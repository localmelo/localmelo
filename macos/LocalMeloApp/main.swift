import AppKit
import Darwin
import Foundation
import WebKit

// The port is fixed for this development build. Tests may choose a different
// non-reserved port via the environment, without changing the user's config.
let defaultPort = 8401

final class App: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var statusItem: NSStatusItem!
    private var child: Process?
    private var timer: Timer?
    private var deadline = Date()
    private var stopping = false
    private var ready = false
    private var checking = false
    // Instance collision check, not authentication against another local process.
    private let instance = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    private var port = defaultPort
    private var logHandle: FileHandle?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        if let override = ProcessInfo.processInfo.environment["LOCALMELO_APP_PORT"] {
            guard let value = Int(override), (1...65535).contains(value) else {
                fail("Invalid LOCALMELO_APP_PORT.")
                return
            }
            port = value
        }
        createWindow()
        createMenu()
        startGateway()
    }

    private func createWindow() {
        webView = WKWebView(frame: .zero)
        webView.navigationDelegate = self
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1050, height: 720),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "LocalMelo"
        window.contentView = webView
        window.delegate = self
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func createMenu() {
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "Open LocalMelo", action: #selector(openWindow), keyEquivalent: "")
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "Quit LocalMelo", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        let root = NSMenu()
        let entry = NSMenuItem()
        entry.submenu = appMenu
        root.addItem(entry)
        NSApp.mainMenu = root
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusItem.button?.title = "Melo"
        statusItem.menu = appMenu.copy() as? NSMenu
    }

    @objc private func openWindow() {
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        sender.orderOut(nil)
        return false
    }

    // lsof identifies the kernel's listener, not just a response at that port.
    private func listenerPIDs(completion: @escaping ([Int32]?) -> Void) {
        let port = self.port
        DispatchQueue.global(qos: .utility).async {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof")
            process.arguments = ["-nP", "-t", "-iTCP:\(port)", "-sTCP:LISTEN"]
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = FileHandle.nullDevice
            do { try process.run() } catch {
                DispatchQueue.main.async { completion(nil) }
                return
            }
            let limit = Date().addingTimeInterval(2)
            while process.isRunning && Date() < limit { usleep(20_000) }
            if process.isRunning {
                kill(process.processIdentifier, SIGKILL)
                process.waitUntilExit()
                DispatchQueue.main.async { completion(nil) }
                return
            }
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            let pids: [Int32]? = process.terminationStatus == 0 || process.terminationStatus == 1
                ? String(decoding: data, as: UTF8.self).split(whereSeparator: \.isNewline).compactMap { Int32($0) }
                : nil
            DispatchQueue.main.async { completion(pids) }
        }
    }

    private func startGateway() {
        listenerPIDs { [weak self] listeners in self?.startGateway(listeners: listeners) }
    }

    private func startGateway(listeners: [Int32]?) {
        guard let listeners = listeners else {
            fail("Cannot inspect port \(port). The gateway was not started.")
            return
        }
        guard listeners.isEmpty else {
            fail("Port \(port) is already in use by process \(listeners.map(String.init).joined(separator: ", ")). Stop the existing gateway or choose another port.")
            return
        }
        guard let executable = Bundle.main.resourceURL?.appendingPathComponent("runtime/bin/melo") else {
            fail("The bundled gateway is missing.")
            return
        }
        let home = URL(fileURLWithPath: ProcessInfo.processInfo.environment["HOME"] ?? NSHomeDirectory(), isDirectory: true)
        let logs = home.appendingPathComponent("Library/Logs/LocalMelo", isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: logs, withIntermediateDirectories: true)
            let path = logs.appendingPathComponent("gateway.log")
            if !FileManager.default.fileExists(atPath: path.path) {
                guard FileManager.default.createFile(atPath: path.path, contents: nil, attributes: [.posixPermissions: 0o600]) else {
                    throw CocoaError(.fileWriteUnknown)
                }
            }
            let handle = try FileHandle(forWritingTo: path)
            try handle.seekToEnd()
            logHandle = handle
        } catch {
            fail("Cannot open the gateway log: \(error.localizedDescription)")
            return
        }
        let process = Process()
        process.executableURL = executable
        process.arguments = ["--serve", "--allow-unconfigured", "--host", "127.0.0.1", "--port", String(port)]
        var environment = ProcessInfo.processInfo.environment
        environment["LOCALMELO_APP_INSTANCE"] = instance
        process.environment = environment
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = logHandle
        process.standardError = logHandle
        process.terminationHandler = { [weak self] process in
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.timer?.invalidate()
                self.tearDownWebView()
                guard !self.stopping else { return }
                self.listenerPIDs { pids in
                    let holders = (pids ?? []).filter { $0 != process.processIdentifier }
                    let detail = holders.isEmpty ? "" : " Port \(self.port) is held by process \(holders.map(String.init).joined(separator: ", "))."
                    self.fail("The gateway exited (status \(process.terminationStatus)).\(detail) See ~/Library/Logs/LocalMelo/gateway.log.")
                }
            }
        }
        do { try process.run() } catch {
            fail("Cannot start the bundled gateway: \(error.localizedDescription)")
            return
        }
        child = process
        deadline = Date().addingTimeInterval(20)
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in self?.checkReady() }
    }

    private func checkReady() {
        guard let child = child, child.isRunning else { return }
        if Date() >= deadline {
            timer?.invalidate()
            stopping = true
            kill(child.processIdentifier, SIGTERM)
            fail("Gateway did not become ready on port \(port). See ~/Library/Logs/LocalMelo/gateway.log.")
            return
        }
        guard !checking else { return }
        checking = true
        listenerPIDs { [weak self] pids in
            guard let self = self else { return }
            guard !self.stopping, child.isRunning, ownsListener(pids, child: child.processIdentifier) else {
                self.checking = false
                return
            }
            self.fetchHealth(child: child)
        }
    }

    private func fetchHealth(child: Process) {
        let url = URL(string: "http://127.0.0.1:\(port)/api/health")!
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        URLSession.shared.dataTask(with: request) { [weak self] data, response, _ in
            DispatchQueue.main.async {
                guard let self = self else { return }
                self.checking = false
                guard let data = data, (response as? HTTPURLResponse)?.statusCode == 200,
                      let health = gatewayHealth(data), !self.ready, !self.stopping,
                      child.isRunning else { return }
                self.listenerPIDs { [weak self] pids in
                    guard let self = self, !self.ready, !self.stopping, child.isRunning,
                          ownsListener(pids, child: child.processIdentifier) else { return }
                    guard health.store == "ok" else {
                        self.timer?.invalidate()
                        self.stopping = true
                        kill(child.processIdentifier, SIGTERM)
                        self.fail("The gateway session store is unavailable: \(health.error). Stop the other gateway and try again.")
                        return
                    }
                    self.ready = true
                    self.timer?.invalidate()
                    self.webView.load(URLRequest(url: URL(string: "http://127.0.0.1:\(self.port)/")!))
                }
            }
        }.resume()
    }

    private func tearDownWebView() {
        guard let view = webView else { return }
        view.stopLoading()
        view.navigationDelegate = nil
        window.contentView = NSView()
        webView = nil
    }

    func webView(_ webView: WKWebView, decidePolicyFor response: WKNavigationResponse,
                 decisionHandler: @escaping (WKNavigationResponsePolicy) -> Void) {
        guard response.isForMainFrame else { decisionHandler(.allow); return }
        guard let http = response.response as? HTTPURLResponse,
              let url = http.url, isGatewayURL(url, port: port),
              acceptsInstance(http.value(forHTTPHeaderField: "X-LocalMelo-Instance"),
                              expected: instance, childRunning: child?.isRunning == true) else {
            decisionHandler(.cancel)
            tearDownWebView()
            fail("Port \(port) is held by another process. Stop the existing listener and try again.")
            return
        }
        decisionHandler(.allow)
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = action.request.url else { decisionHandler(.cancel); return }
        if url.scheme == "http", url.host == "127.0.0.1", url.port == port {
            decisionHandler(.allow)
        } else {
            decisionHandler(.cancel)
            if action.navigationType == .linkActivated, ["http", "https"].contains(url.scheme ?? "") {
                NSWorkspace.shared.open(url)
            }
        }
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let child = child, child.isRunning else { return .terminateNow }
        if stopping {
            kill(child.processIdentifier, SIGTERM)
            waitForChild(child, application: sender)
            return .terminateLater
        }
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/health")!)
        request.timeoutInterval = 2
        URLSession.shared.dataTask(with: request) { [weak self] data, response, _ in
            var running: Int?
            if let data = data, (response as? HTTPURLResponse)?.statusCode == 200,
               let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let sessions = object["sessions"] as? [String: Any] {
                running = sessions["running"] as? Int
            }
            DispatchQueue.main.async {
                guard let self = self else { sender.reply(toApplicationShouldTerminate: false); return }
                // Always ask: a turn can start while the health request is in flight.
                do {
                    let alert = NSAlert()
                    alert.messageText = "Quit LocalMelo?"
                    alert.informativeText = running == nil
                        ? "The task status is unavailable. Quitting stops the gateway and any running tasks."
                        : "Quitting stops the gateway and any running tasks."
                    alert.addButton(withTitle: "Quit and Stop")
                    alert.addButton(withTitle: "Cancel")
                    guard alert.runModal() == .alertFirstButtonReturn else {
                        sender.reply(toApplicationShouldTerminate: false)
                        return
                    }
                }
                self.stopping = true
                self.timer?.invalidate()
                if child.isRunning { kill(child.processIdentifier, SIGTERM) }
                self.waitForChild(child, application: sender)
            }
        }.resume()
        return .terminateLater
    }

    private func waitForChild(_ child: Process, application: NSApplication) {
        DispatchQueue.global().async {
            child.waitUntilExit()
            DispatchQueue.main.async { application.reply(toApplicationShouldTerminate: true) }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 8) {
            if child.isRunning { kill(child.processIdentifier, SIGKILL) }
        }
    }

    private func fail(_ text: String) {
        stopping = true
        timer?.invalidate()
        tearDownWebView()
        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = "LocalMelo could not start"
        alert.informativeText = text
        alert.runModal()
        NSApp.terminate(nil)
    }
}

let application = NSApplication.shared
let delegate = App()
application.delegate = delegate
application.run()
