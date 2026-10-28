import AppKit
import Darwin
import Foundation
import WebKit

// The port is fixed for this development build. Tests may choose a different
// non-reserved port via the environment, without changing the user's config.
let defaultPort = 8401

final class ConsoleWebView: WKWebView {
    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        guard event.modifierFlags.intersection(.deviceIndependentFlagsMask).contains(.command),
              let key = event.charactersIgnoringModifiers?.lowercased() else {
            return super.performKeyEquivalent(with: event)
        }
        let actions = ["c": "copy:", "v": "paste:", "x": "cut:", "a": "selectAll:",
                       "z": event.modifierFlags.contains(.shift) ? "redo:" : "undo:"]
        if let name = actions[key], NSApp.sendAction(Selector(name), to: nil, from: self) { return true }
        return super.performKeyEquivalent(with: event)
    }
}

final class App: NSObject, NSApplicationDelegate, NSWindowDelegate, WKNavigationDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var statusItem: NSStatusItem!
    private var child: Process?
    private var timer: Timer?
    private var deadline = Date()
    private var stopping = false
    private var quitPending = false
    private var ready = false
    private var checking = false
    private let healthSession = URLSession(configuration: .ephemeral)
    private var healthAuthenticated = false
    private var browserLoginCode: String?
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
        webView = ConsoleWebView(frame: .zero)
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
        let edit = NSMenu(title: "Edit")
        for (title, action, key) in [("Undo", "undo:", "z"), ("Redo", "redo:", "Z"),
                                      ("Cut", "cut:", "x"), ("Copy", "copy:", "c"),
                                      ("Paste", "paste:", "v"), ("Select All", "selectAll:", "a")] {
            edit.addItem(withTitle: title, action: Selector(action), keyEquivalent: key)
        }
        let editEntry = NSMenuItem()
        editEntry.submenu = edit
        root.addItem(editEntry)
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
            fail("Port \(port) is already in use by process \(listeners.map(String.init).joined(separator: ", ")). Stop the listener or choose another port. For the melo service, use melo --daemon status and melo --daemon uninstall.")
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
            RunLoop.main.perform(inModes: [.default, .modalPanel]) { [weak self] in
                guard let self = self else { return }
                self.timer?.invalidate()
                self.tearDownWebView()
                guard !self.stopping, !self.quitPending else { return }
                self.listenerPIDs { pids in
                    let holders = (pids ?? []).filter { $0 != process.processIdentifier }
                    let detail = holders.isEmpty ? "" : " Port \(self.port) is held by process \(holders.map(String.init).joined(separator: ", "))."
                    let exit = gatewayExitMessage(status: process.terminationStatus, signalled: process.terminationReason == .uncaughtSignal)
                    self.fail("\(exit)\(detail) See ~/Library/Logs/LocalMelo/gateway.log.", title: self.ready ? "LocalMelo gateway stopped" : "LocalMelo could not start")
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
            if self.healthAuthenticated {
                self.fetchHealth(child: child)
            } else {
                self.authenticateHealth(child: child)
            }
        }
    }

    private func authenticateHealth(child: Process) {
        guard let python = Bundle.main.resourceURL?.appendingPathComponent("runtime/bin/python/bin/python3") else {
            checking = false
            return
        }
        let port = self.port
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let process = Process()
            process.executableURL = python
            process.arguments = ["-P", "-c", "from localmelo.support.gateway.auth import GatewayAuth; import sys; a=GatewayAuth(); print(a.issue_login(int(sys.argv[1]))); print(a.issue_login(int(sys.argv[1]))); a.close()", String(port)]
            var env = ProcessInfo.processInfo.environment
            env.removeValue(forKey: "PYTHONPATH")
            env.removeValue(forKey: "PYTHONHOME")
            env["PYTHONNOUSERSITE"] = "1"
            process.environment = env
            process.standardInput = FileHandle.nullDevice
            process.standardError = FileHandle.nullDevice
            let output = Pipe()
            process.standardOutput = output
            do { try process.run() } catch {
                DispatchQueue.main.async { self?.checking = false }
                return
            }
            let limit = Date().addingTimeInterval(3)
            while process.isRunning && Date() < limit { usleep(20_000) }
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
            process.waitUntilExit()
            let data = output.fileHandleForReading.readDataToEndOfFile()
            let codes = String(decoding: data, as: UTF8.self).split(whereSeparator: \.isNewline).map(String.init)
            guard process.terminationStatus == 0, codes.count == 2,
                  codes.allSatisfy({ $0.count <= 256 }) else {
                DispatchQueue.main.async { self?.checking = false }
                return
            }
            let url = URL(string: "http://127.0.0.1:\(port)/api/auth/login")!
            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.timeoutInterval = 2
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.setValue("http://127.0.0.1:\(port)", forHTTPHeaderField: "Origin")
            request.httpBody = try? JSONSerialization.data(withJSONObject: ["code": codes[0]])
            self?.healthSession.dataTask(with: request) { [weak self] _, response, _ in
                DispatchQueue.main.async {
                    guard let self = self else { return }
                    self.checking = false
                    guard !self.stopping, child.isRunning,
                          (response as? HTTPURLResponse)?.statusCode == 204 else { return }
                    self.healthAuthenticated = true
                    self.browserLoginCode = codes[1]
                }
            }.resume()
        }
    }

    private func fetchHealth(child: Process) {
        let url = URL(string: "http://127.0.0.1:\(port)/api/health")!
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        healthSession.dataTask(with: request) { [weak self] data, response, _ in
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
                        self.fail(storeFailureMessage(health))
                        return
                    }
                    guard let code = self.browserLoginCode,
                          let url = URL(string: "http://127.0.0.1:\(self.port)/#login=\(code)") else { return }
                    self.browserLoginCode = nil
                    self.ready = true
                    self.timer?.invalidate()
                    self.webView.load(URLRequest(url: url))
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
        guard let http = response.response as? HTTPURLResponse, let url = http.url else {
            decisionHandler(.cancel)
            return
        }
        guard isGatewayPage(url, port: port) else {
            decisionHandler(.cancel)
            if child?.isRunning == true {
                webView.load(URLRequest(url: URL(string: "http://127.0.0.1:\(port)/")!))
            }
            return
        }
        guard acceptsInstance(http.value(forHTTPHeaderField: "X-LocalMelo-Instance"),
                              expected: instance, childRunning: child?.isRunning == true) else {
            decisionHandler(.cancel)
            tearDownWebView()
            fail("Port \(port) is held by another process. Stop the existing listener and try again.")
            return
        }
        decisionHandler(.allow)
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        guard child?.isRunning == true else {
            tearDownWebView()
            return
        }
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = action.request.url else { decisionHandler(.cancel); return }
        if isGatewayURL(url, port: port) {
            decisionHandler(action.targetFrame?.isMainFrame == false || isGatewayPage(url, port: port) ? .allow : .cancel)
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
        if quitPending { return .terminateCancel }
        quitPending = true
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/health")!)
        request.timeoutInterval = 2
        healthSession.dataTask(with: request) { [weak self] data, response, _ in
            var running: Int?
            if let data = data, (response as? HTTPURLResponse)?.statusCode == 200,
               let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let sessions = object["sessions"] as? [String: Any] {
                running = sessions["running"] as? Int
            }
            RunLoop.main.perform(inModes: [.default, .modalPanel]) {
                guard let self = self else { sender.reply(toApplicationShouldTerminate: false); return }
                guard child.isRunning else {
                    self.stopping = true
                    self.tearDownWebView()
                    sender.reply(toApplicationShouldTerminate: true)
                    return
                }
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
                        self.quitPending = false
                        if !child.isRunning {
                            self.tearDownWebView()
                            sender.reply(toApplicationShouldTerminate: false)
                            DispatchQueue.main.async {
                                self.fail("The gateway exited while the quit confirmation was open.")
                            }
                        } else {
                            sender.reply(toApplicationShouldTerminate: false)
                        }
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
            RunLoop.main.perform(inModes: [.default, .modalPanel]) {
                application.reply(toApplicationShouldTerminate: true)
            }
        }
        let timeout = Timer(timeInterval: 8, repeats: false) { _ in
            if child.isRunning { kill(child.processIdentifier, SIGKILL) }
        }
        RunLoop.main.add(timeout, forMode: .default)
        RunLoop.main.add(timeout, forMode: .modalPanel)
    }

    private func fail(_ text: String, title: String = "LocalMelo could not start") {
        stopping = true
        timer?.invalidate()
        tearDownWebView()
        let alert = NSAlert()
        alert.alertStyle = .critical
        alert.messageText = title
        alert.informativeText = text
        alert.runModal()
        NSApp.terminate(nil)
    }
}

let application = NSApplication.shared
let delegate = App()
application.delegate = delegate
application.run()
