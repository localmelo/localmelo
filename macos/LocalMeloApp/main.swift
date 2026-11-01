import AppKit
import Darwin
import Foundation
import WebKit

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
    private var statusView: StatusView!
    private var menuBar: MenuBar!
    // Melo on the desktop while the window is put away, and the activity pill that takes the pet's place while work runs.
    private var pet: PetController!
    private var pill: PillController!
    // Where "Waiting for your approval" goes: the console route of the first chat or task that waits.
    private var pendingRoute: String?
    // The app's own gateway, or a running one it attached to (see `GatewayChild.owned`).
    private var child: GatewayChild?
    // Counts starts, so an endpoint probe from an earlier Try Again does nothing.
    private var startAttempt = 0
    private var quitItems: [NSMenuItem] = []
    private var timer: Timer?
    private var healthTimer: Timer?
    private var deadline = Date()
    private var stopping = false
    private var quitPending = false
    private var ready = false
    private var checking = false
    private var nextLoginAttempt = Date()
    private let healthSession = URLSession(configuration: .ephemeral)
    private var healthAuthenticated = false
    private var browserLoginCode: String?
    private var signInAttempts = 0
    // What decides whether the menus offer "Sign In Again": attempts that failed in a row, and what the page says.
    private var signInFailures = 0
    private var page = PageState()
    private var signInItems: [NSMenuItem] = []
    // Instance collision check, not authentication against another local process.
    private let ownInstance = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
    // The instance of the gateway the app talks to: its own, or the attached one's (empty
    // when the terminal started it).
    private var instance = ""
    // The gateway picks its own port (`--port 0`) and prints it; 0 until then.
    private var port = 0
    private var logHandle: FileHandle?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // The console's language, before any word is shown: the setting, else the Mac's languages.
        let home = URL(fileURLWithPath: ProcessInfo.processInfo.environment["HOME"] ?? NSHomeDirectory(), isDirectory: true)
        let config = try? String(contentsOf: home.appendingPathComponent(".cache/localmelo/config.toml"), encoding: .utf8)
        L10n.use(pickLanguage(setting: consoleLanguageSetting(config), preferred: Locale.preferredLanguages))
        NSApp.setActivationPolicy(.regular)
        createWindow()
        createMenu()
        createPet()
        startGateway()
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        if !window.isVisible || window.isMiniaturized { openWindow() }
        return false
    }

    func applicationDidHide(_ notification: Notification) { updatePet() }
    func applicationDidUnhide(_ notification: Notification) { updatePet() }
    func windowDidMiniaturize(_ notification: Notification) { updatePet() }
    func windowDidDeminiaturize(_ notification: Notification) { updatePet() }

    private func createWindow() {
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1280, height: 800),
                          styleMask: [.titled, .closable, .miniaturizable, .resizable],
                          backing: .buffered, defer: false)
        window.title = "localmelo"
        window.backgroundColor = Palette.canvas
        // The status view covers the window until the console has loaded, and comes back
        // when it cannot: the window never shows an empty web view.
        let content = NSView(frame: NSRect(x: 0, y: 0, width: 1280, height: 800))
        statusView = StatusView(pictures: Bundle.main.resourceURL?.appendingPathComponent("moves"))
        statusView.frame = content.bounds
        statusView.autoresizingMask = [.width, .height]
        statusView.onAction = { [weak self] action in self?.handle(action) }
        content.addSubview(statusView)
        window.contentView = content
        window.delegate = self
        window.center()
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func createMenu() {
        let appMenu = NSMenu()
        appMenu.addItem(withTitle: "", action: #selector(openWindow), keyEquivalent: "").word(.open)
        appMenu.addItem(withTitle: "", action: #selector(openSettings), keyEquivalent: ",").word(.settings)
        let signIn = appMenu.addItem(withTitle: "", action: #selector(signInAgain), keyEquivalent: "").word(.signIn)
        appMenu.addItem(.separator())
        appMenu.addItem(withTitle: "", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q").word(.quit)
        let root = NSMenu()
        let entry = NSMenuItem()
        entry.submenu = appMenu
        root.addItem(entry)
        let edit = NSMenu()
        for (word, action, key) in [(MenuWord.undo, "undo:", "z"), (.redo, "redo:", "Z"), (.cut, "cut:", "x"), (.copy, "copy:", "c"),
                                    (.paste, "paste:", "v"), (.selectAll, "selectAll:", "a")] {
            edit.addItem(withTitle: "", action: Selector(action), keyEquivalent: key).word(word)
        }
        let editEntry = NSMenuItem()
        editEntry.submenu = edit
        editEntry.word(.edit)
        root.addItem(editEntry)
        NSApp.mainMenu = root
        menuBar = MenuBar(menu: appMenu.copy() as? NSMenu ?? NSMenu(), waiting: #selector(openWaiting),
                          showPet: #selector(showPet))
        signInItems = [signIn] + (menuBar.statusItem.menu?.items.filter { $0.action == #selector(signInAgain) } ?? [])
        let quit = #selector(NSApplication.terminate(_:))
        quitItems = appMenu.items.filter { $0.action == quit }
            + (menuBar.statusItem.menu?.items.filter { $0.action == quit } ?? [])
        updateSignInItems()
    }

    private func createPet() {
        let home = URL(fileURLWithPath: ProcessInfo.processInfo.environment["HOME"] ?? NSHomeDirectory(), isDirectory: true)
        let moves = Bundle.main.resourceURL?.appendingPathComponent("moves")
        pet = PetController(settingsURL: home.appendingPathComponent("Library/Application Support/LocalMelo/pet.json"),
                            movesFolder: moves)
        pet.openWindow = { [weak self] in self?.openWindow() }
        pet.openWaiting = { [weak self] in self?.openWaiting() }
        pet.hiddenChanged = { [weak self] hidden in self?.menuBar.showPetLine.isHidden = !hidden }
        menuBar.showPetLine.isHidden = !pet.hiddenByPerson
        menuBar.offerDesktop(pet.desktop.makeItem())
        pill = PillController(pictures: moves,
                              settings: { [weak self] in self?.pet.settings ?? PetSettings() },
                              remember: { [weak self] key, spot in self?.pet.rememberSpot(screen: key, spot: spot) })
        pill.openWindow = { [weak self] in self?.openWindow() }
        pill.openWaiting = { [weak self] in self?.openWaiting() }
        pill.openRoute = { [weak self] route in self?.open(route: route) }
        pill.view.menu = pet.view.menu
        pill.showingChanged = { [weak self] showing, done in self?.pet.pillShowing(showing, showsDone: done) }
        pet.desktopChanged = { [weak self] mode in self?.pill.desktop(mode) }
        pill.desktop(pet.settings.mode)
        updatePet()
    }

    // The pet lies on the desktop while the window is closed, minimized or the app hidden.
    private func updatePet() {
        let shown = petWindowShown(visible: window.isVisible, minimized: window.isMiniaturized, appHidden: NSApp.isHidden)
        // The pet hears first when the window comes back and last when it goes away, so neither flashes.
        let hear: [(Bool) -> Void] = [{ self.pet?.windowShown($0) }, { self.pill?.windowShown($0) }]
        (shown ? hear : hear.reversed()).forEach { $0(shown) }
    }

    @objc private func showPet() { pet.setHidden(false) }

    // The console page names the language it chose each time it loads, so also after Settings changed it: every menu, the
    // pill and the pet's label follow without the app starting over. The same language again does nothing.
    private func languageChosen(_ language: String) {
        guard language != L10n.language else { return }
        L10n.use(language)
        let attached = child.map { !$0.owned } ?? false
        for menu in [NSApp.mainMenu, menuBar.statusItem.menu, pet.view.menu].compactMap({ $0 }) { retitle(menu, attached: attached) }
        menuBar.refresh()
        pet.languageChanged()
        pill.languageChanged()
    }

    private func handle(_ action: FailureAction) {
        switch action {
        case .tryAgain: startGateway()
        case .openLog: openLog()
        case .quit: NSApp.terminate(nil)
        }
    }

    private var logsDirectory: URL {
        let home = URL(fileURLWithPath: ProcessInfo.processInfo.environment["HOME"] ?? NSHomeDirectory(), isDirectory: true)
        return home.appendingPathComponent("Library/Logs/LocalMelo", isDirectory: true)
    }

    private func openLog() {
        let log = logsDirectory.appendingPathComponent("gateway.log")
        let target = FileManager.default.fileExists(atPath: log.path) ? log : logsDirectory
        if !NSWorkspace.shared.open(target) { NSSound.beep() }
    }

    private func showStep() {
        statusView.showStarting(step: startStepText(portKnown: port != 0, ready: ready), ready: ready)
    }

    private func showConsole() {
        statusView.hide()
        window.makeFirstResponder(webView)
    }

    @objc private func openWindow() {
        if NSApp.isHidden { NSApp.unhide(nil) }
        if window.isMiniaturized { window.deminiaturize(nil) }
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        updatePet()
    }

    // The console has no route for its settings, so this presses the button its sidebar has; it needs
    // the console, and before it has loaded the window just opens.
    @objc private func openSettings() {
        openWindow()
        webView?.evaluateJavaScript("document.getElementById('settings-btn')?.click()")
    }

    // The window, then the chat or task that waits. The route reaches the page as an argument, never as
    // script text, and the console follows the hash as it does for a link. A page already at that address
    // gets no event from setting it again (a task shown in a panel the window has since narrowed past), so
    // it is told the same way.
    @objc private func openWaiting() { open(route: pendingRoute) }

    private func open(route: String?) {
        openWindow()
        guard let route = route else { return }
        let script = "if (location.hash === route) { dispatchEvent(new HashChangeEvent('hashchange')); } else { location.hash = route; }"
        webView?.callAsyncJavaScript(script, arguments: ["route": route], in: nil, in: .page)
    }

    // "Sign In Again" shows only after sign-ins failed or while the console is locked, and in developer mode.
    private func updateSignInItems() {
        let show = offersSignIn(failures: signInFailures, page: page)
        for item in signInItems { item.isHidden = !show }
    }

    private func refreshPageState() {
        webView?.evaluateJavaScript(pageStateScript) { [weak self] value, _ in
            guard let self = self else { return }
            self.page = pageState(value as? String)
            self.updateSignInItems()
        }
    }

    // A manual sign-in uses the same instance-checked flow as startup and
    // rotation recovery. It issues fresh one-use codes.
    @objc private func signInAgain() {
        guard !stopping, !checking, let child = child, child.isRunning else { return }
        healthAuthenticated = false
        browserLoginCode = nil
        nextLoginAttempt = Date()
        if ready {
            pollHealth()
        } else {
            checkReady()
        }
    }

    func windowShouldClose(_ sender: NSWindow) -> Bool {
        sender.orderOut(nil)
        updatePet()
        return false
    }

    // lsof identifies the kernel's listener, not just a response at that port.
    private func listenerPIDs(completion: @escaping ([Int32]?) -> Void) {
        let port = self.port
        guard port != 0 else { return completion([]) }
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

    // A probe response can arrive just before another process takes the port,
    // so check the kernel's listener again immediately before a code leaves
    // the app: a listener the app's child does not own must receive none.
    private func sendWhenOwned(child: GatewayChild, _ send: @escaping () -> Void) {
        listenerPIDs { [weak self] pids in
            guard let self = self, !self.stopping else { return }
            guard child.isRunning else {
                self.checking = false
                return
            }
            guard let pids = pids else {
                self.checking = false
                self.fail(L10n.t("app.fail.inspectPort", "Cannot inspect port {port}. No login code was sent.", ["port": String(self.port)]))
                return
            }
            guard ownsListener(pids, child: child.processIdentifier) else {
                self.checking = false
                self.fail(portHeldMessage(self.port))
                return
            }
            send()
        }
    }

    // Also what Try Again runs: a new child, with nothing kept from an earlier one.
    // A gateway already running (the terminal's, say) holds the main chat's store, so
    // the app attaches to it when the endpoint file names a live one, and starts its
    // own only when none runs.
    private func startGateway() {
        startAttempt += 1
        let attempt = startAttempt
        resetForStart()
        let home = URL(fileURLWithPath: ProcessInfo.processInfo.environment["HOME"] ?? NSHomeDirectory(), isDirectory: true)
        let file = home.appendingPathComponent(".cache/localmelo/gateway/endpoint.json")
        let found = gatewayEndpoint(try? Data(contentsOf: file))
        guard case .attach(let endpoint) = gatewayStart(found, alive: { kill($0, 0) == 0 || errno == EPERM }),
              let url = URL(string: "http://127.0.0.1:\(endpoint.port)/") else {
            spawnGateway()
            return
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        healthSession.dataTask(with: request) { [weak self] _, response, _ in
            let http = response as? HTTPURLResponse
            let confirmed = attachConfirmed(status: http?.statusCode,
                                            gatewayHeader: http?.value(forHTTPHeaderField: "X-LocalMelo-Gateway"),
                                            endpoint: endpoint)
            DispatchQueue.main.async {
                guard let self = self, self.startAttempt == attempt, !self.quitPending else { return }
                if confirmed {
                    self.attach(endpoint)
                } else {
                    self.spawnGateway()
                }
            }
        }.resume()
    }

    // Signs in to a running gateway the app did not start, the same way as to its own:
    // the listener check, the instance check, and one-use codes from the shared auth folder.
    private func attach(_ endpoint: GatewayEndpoint) {
        instance = endpoint.instance
        port = endpoint.port
        child = AttachedGateway(pid: endpoint.pid)
        setQuitTitles(attached: true)
        showStep()
        deadline = Date().addingTimeInterval(20)
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in self?.checkReady() }
    }

    private func setQuitTitles(attached: Bool) {
        for item in quitItems { item.title = quitTitle(attached: attached) }
    }

    private func resetForStart() {
        stopping = false
        ready = false
        checking = false
        healthAuthenticated = false
        browserLoginCode = nil
        nextLoginAttempt = Date()
        signInFailures = 0
        updateSignInItems()
        pet?.update(health: nil, stopped: false)
        pill?.update(health: nil, stopped: false)
        port = 0
        instance = ownInstance
        setQuitTitles(attached: false)
        showStep()
    }

    private func spawnGateway() {
        guard let executable = Bundle.main.resourceURL?.appendingPathComponent("runtime/bin/melo") else {
            fail(L10n.t("app.fail.missing", "The bundled gateway is missing."))
            return
        }
        let logs = logsDirectory
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
            fail(L10n.t("app.fail.log", "Cannot open the gateway log: {error}", ["error": error.localizedDescription]))
            return
        }
        let process = Process()
        process.executableURL = executable
        process.arguments = ["--serve", "--allow-unconfigured", "--host", "127.0.0.1", "--port", "0"]
        var environment = ProcessInfo.processInfo.environment
        environment["LOCALMELO_APP_INSTANCE"] = ownInstance
        process.environment = environment
        process.standardInput = FileHandle.nullDevice
        let output = Pipe()
        process.standardOutput = output
        process.standardError = logHandle
        process.terminationHandler = { [weak self] process in
            RunLoop.main.perform(inModes: [.default, .modalPanel]) { [weak self] in
                guard let self = self, self.child === process else { return }
                self.timer?.invalidate()
                self.tearDownWebView()
                guard !self.stopping, !self.quitPending else { return }
                self.listenerPIDs { pids in
                    let holders = (pids ?? []).filter { $0 != process.processIdentifier }
                    let detail = holders.isEmpty ? "" : L10n.t("app.fail.portHolders", " Port {port} is held by process {pids}.",
                                                               ["port": String(self.port), "pids": holders.map(String.init).joined(separator: ", ")])
                    let log = gatewayExitLogLine(status: process.terminationStatus, signalled: process.terminationReason == .uncaughtSignal)
                    try? self.logHandle?.write(contentsOf: Data("\n\(log)\(detail)\n".utf8))
                    self.fail(gatewayExitMessage(detail: detail))
                }
            }
        }
        do { try process.run() } catch {
            fail(L10n.t("app.fail.start", "Cannot start the bundled gateway: {error}", ["error": error.localizedDescription]))
            return
        }
        readStartup(output, of: process)
        child = process
        deadline = Date().addingTimeInterval(20)
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in self?.checkReady() }
    }

    // The gateway binds a free port first and then prints "Gateway: http://HOST:PORT".
    // Keep reading after that line so the child never blocks on a full pipe, and
    // write everything to the log, where its output went before.
    private func readStartup(_ output: Pipe, of process: Process) {
        let log = logHandle
        var startup = StartupOutput()
        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else {
                handle.readabilityHandler = nil
                return
            }
            try? log?.write(contentsOf: data)
            if let port = startup.append(data) {
                DispatchQueue.main.async {
                    guard let self = self, self.child === process, self.port == 0 else { return }
                    self.port = port
                    self.showStep()
                }
            }
        }
    }

    private func checkReady() {
        guard let child = child else { return }
        guard child.isRunning else {
            if !child.owned { fail(attachedStoppedMessage) }
            return
        }
        if Date() >= deadline {
            timer?.invalidate()
            stopping = true
            if child.owned { kill(child.processIdentifier, SIGTERM) }
            fail(port == 0 ? L10n.t("app.fail.noPort", "The gateway did not report its port.")
                 : L10n.t("app.fail.notReady", "Gateway did not become ready on port {port}.", ["port": String(port)]))
            return
        }
        guard port != 0, !checking else { return }
        if !healthAuthenticated && Date() < nextLoginAttempt { return }
        checking = true
        listenerPIDs { [weak self] pids in
            guard let self = self else { return }
            guard !self.stopping, child.isRunning, ownsListener(pids, child: child.processIdentifier) else {
                self.checking = false
                return
            }
            if self.healthAuthenticated {
                self.fetchHealth(child: child, reload: false)
            } else {
                self.signIn(child: child, reload: false)
            }
        }
    }

    // Prove that the listener is this child before issuing any login code.
    // The identifier catches an accidental collision; it is not authentication.
    private func signIn(child: GatewayChild, reload: Bool) {
        let url = URL(string: "http://127.0.0.1:\(port)/")!
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        request.setValue(instance, forHTTPHeaderField: "X-LocalMelo-Instance")
        healthSession.dataTask(with: request) { [weak self] _, response, _ in
            DispatchQueue.main.async {
                guard let self = self else { return }
                guard !self.stopping, child.isRunning else {
                    self.checking = false
                    return
                }
                guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                    self.retryLogin()
                    return
                }
                guard acceptsInstance(http.value(forHTTPHeaderField: "X-LocalMelo-Instance"),
                                      expected: self.instance, childRunning: child.isRunning) else {
                    self.checking = false
                    self.fail(portHeldMessage(self.port))
                    return
                }
                self.issueLoginCodes(child: child, reload: reload)
            }
        }.resume()
    }

    private func issueLoginCodes(child: GatewayChild, reload: Bool) {
        guard let python = Bundle.main.resourceURL?.appendingPathComponent("runtime/bin/python/bin/python3") else {
            retryLogin()
            return
        }
        let port = self.port
        let instance = self.instance
        DispatchQueue.global(qos: .utility).async { [weak self] in
            let process = Process()
            process.executableURL = python
            process.arguments = ["-P", "-c", "from localmelo.support.gateway.auth import GatewayAuth; import sys; a=GatewayAuth(); print(a.issue_login(int(sys.argv[1]), instance=sys.argv[2])); print(a.issue_login(int(sys.argv[1]), instance=sys.argv[2])); a.close()", String(port), instance]
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
                DispatchQueue.main.async { self?.retryLogin() }
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
                DispatchQueue.main.async { self?.retryLogin() }
                return
            }
            DispatchQueue.main.async {
                guard let self = self else { return }
                self.sendWhenOwned(child: child) {
                    let url = URL(string: "http://127.0.0.1:\(port)/api/auth/login")!
                    var request = URLRequest(url: url)
                    request.httpMethod = "POST"
                    request.timeoutInterval = 2
                    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    request.setValue("http://127.0.0.1:\(port)", forHTTPHeaderField: "Origin")
                    request.setValue(instance, forHTTPHeaderField: "X-LocalMelo-Instance")
                    request.httpBody = try? JSONSerialization.data(withJSONObject: ["code": codes[0]])
                    self.healthSession.dataTask(with: request) { [weak self] _, response, _ in
                        DispatchQueue.main.async {
                            guard let self = self else { return }
                            guard !self.stopping, child.isRunning,
                                  (response as? HTTPURLResponse)?.statusCode == 204 else {
                                self.retryLogin()
                                return
                            }
                            // The cookie is presumed, not proven: fetchHealth verifies
                            // it and clears the flag on 401 so sign-in runs again.
                            self.healthAuthenticated = true
                            self.browserLoginCode = codes[1]
                            self.fetchHealth(child: child, reload: reload)
                        }
                    }.resume()
                }
            }
        }
    }

    private func retryLogin() {
        checking = false
        nextLoginAttempt = Date().addingTimeInterval(0.5)
        signInFailures += 1
        updateSignInItems()
    }

    private func fetchHealth(child: GatewayChild, reload: Bool) {
        let url = URL(string: "http://127.0.0.1:\(port)/api/health")!
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        // What the gateway writes for the pill and the menus, in the app's language.
        request.setValue(L10n.language, forHTTPHeaderField: "X-LocalMelo-Language")
        healthSession.dataTask(with: request) { [weak self] data, response, _ in
            DispatchQueue.main.async {
                guard let self = self else { return }
                self.checking = false
                guard !self.stopping, child.isRunning else { return }
                let status = (response as? HTTPURLResponse)?.statusCode
                if status == 401 {
                    self.healthAuthenticated = false
                    self.nextLoginAttempt = Date().addingTimeInterval(0.5)
                    self.signInFailures += 1
                    self.updateSignInItems()
                    return
                }
                guard status == 200, let data = data, let health = gatewayHealth(data) else { return }
                self.healthAuthenticated = true
                self.signInFailures = 0
                self.menuBar.show(menuBarState(data))
                self.pet.update(health: data, stopped: false)
                self.pill.update(health: data, stopped: false)
                self.pendingRoute = waitingRoute(data)
                self.refreshPageState()
                guard health.store == "ok" else {
                    if !self.ready {
                        self.timer?.invalidate()
                        self.healthTimer?.invalidate()
                        self.stopping = true
                        if child.owned { kill(child.processIdentifier, SIGTERM) }
                        self.fail(storeFailureMessage(health))
                    }
                    return
                }
                guard !self.ready || reload else { return }
                guard let code = self.browserLoginCode else { return }
                self.browserLoginCode = nil
                self.ready = true
                self.timer?.invalidate()
                self.startHealthPolling()
                // A changing query makes a re-sign-in a new document load; a
                // fragment-only change would not run the page's login code again.
                self.signInAttempts += 1
                guard let url = URL(
                    string: "http://127.0.0.1:\(self.port)/?auth=\(self.signInAttempts)#login=\(code)&instance=\(self.instance)"
                ) else { return }
                self.sendWhenOwned(child: child) {
                    self.loadConsole(url)
                }
            }
        }.resume()
    }

    // A fresh web view each time avoids a stalled second navigation that WebKit can
    // leave uncommitted after the first page loaded. It loads under the status view
    // and is shown once the page has loaded.
    private func loadConsole(_ url: URL) {
        if webView != nil { tearDownWebView() }
        guard let content = window.contentView else { return }
        let configuration = WKWebViewConfiguration()
        // The console page picks its language as the app did, from the Mac's languages.
        configuration.userContentController.addUserScript(
            WKUserScript(source: languagesScript(Locale.preferredLanguages), injectionTime: .atDocumentStart,
                         forMainFrameOnly: true))
        configuration.userContentController.addScriptMessageHandler(
            FolderBridge(gatewayPort: { [weak self] in self?.port ?? 0 },
                         choose: { [weak self] reply in self?.chooseFolder(reply) ?? reply(nil) },
                         reveal: { [weak self] path, reply in self?.revealFolder(path, reply) ?? reply(false) }),
            contentWorld: .page, name: FolderBridge.name)
        configuration.userContentController.addScriptMessageHandler(
            DesktopBridge(gatewayPort: { [weak self] in self?.port ?? 0 },
                          current: { [weak self] in self?.pet?.settings.mode ?? .pill },
                          choose: { [weak self] mode in self?.pet?.setDesktop(mode) }),
            contentWorld: .page, name: DesktopBridge.name)
        configuration.userContentController.add(
            QuitBridge(gatewayPort: { [weak self] in self?.port ?? 0 }, quit: { NSApp.terminate(nil) }),
            contentWorld: .page, name: QuitBridge.name)
        configuration.userContentController.add(
            LanguageBridge(gatewayPort: { [weak self] in self?.port ?? 0 }, changed: { [weak self] in self?.languageChosen($0) }),
            contentWorld: .page, name: LanguageBridge.name)
        let view = ConsoleWebView(frame: content.bounds, configuration: configuration)
        view.autoresizingMask = [.width, .height]
        view.navigationDelegate = self
        view.underPageBackgroundColor = Palette.canvas
        content.addSubview(view, positioned: .below, relativeTo: statusView)
        webView = view
        showStep()
        view.load(URLRequest(url: url))
    }

    private func startHealthPolling() {
        healthTimer?.invalidate()
        healthTimer = Timer.scheduledTimer(withTimeInterval: 2, repeats: true) { [weak self] _ in
            self?.pollHealth()
        }
    }

    private func pollHealth() {
        // An attached gateway has no termination handler: its end shows here.
        if let child = child, !child.owned, !child.isRunning, !stopping {
            fail(attachedStoppedMessage)
            return
        }
        guard ready, !stopping, !checking, Date() >= nextLoginAttempt,
              let child = child, child.isRunning else { return }
        checking = true
        if healthAuthenticated {
            fetchHealth(child: child, reload: false)
        } else {
            signIn(child: child, reload: true)
        }
    }

    private func tearDownWebView() {
        webView?.stopLoading()
        webView?.navigationDelegate = nil
        webView?.configuration.userContentController.removeScriptMessageHandler(forName: FolderBridge.name, contentWorld: .page)
        webView?.configuration.userContentController.removeScriptMessageHandler(forName: DesktopBridge.name, contentWorld: .page)
        webView?.configuration.userContentController.removeScriptMessageHandler(forName: LanguageBridge.name, contentWorld: .page)
        webView?.removeFromSuperview()
        webView = nil
        page = PageState()
        updateSignInItems()
    }

    // The folders the person picked in the panel: the page may ask to see them in Finder, and nothing else but a chat's folder.
    private var pickedFolders: [String] = []

    // The system's folder panel, as a sheet on the window; the reply is the path it returns, or nil.
    private func chooseFolder(_ reply: @escaping (String?) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = L10n.t("app.folder.message", "Choose the folder melo works in")
        panel.prompt = L10n.t("app.folder.choose", "Choose")
        panel.beginSheetModal(for: window) { response in
            let path = response == .OK ? panel.url?.path : nil
            if let path = path { self.pickedFolders.append(path) }
            reply(path)
        }
    }

    // Shows a folder in Finder if it exists and is a chat's folder, as the gateway lists them, or one picked in the panel.
    private func revealFolder(_ requested: String, _ reply: @escaping (Bool) -> Void) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/api/sessions")!)
        request.timeoutInterval = 2
        healthSession.dataTask(with: request) { [weak self] data, response, _ in
            DispatchQueue.main.async {
                let listed = (response as? HTTPURLResponse)?.statusCode == 200 ? data.map(chatFolders) ?? [] : []
                guard let self = self, let url = revealTarget(requested, allowed: listed + self.pickedFolders) else {
                    reply(false)
                    return
                }
                NSWorkspace.shared.activateFileViewerSelecting([url])
                reply(true)
            }
        }.resume()
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
            fail(portHeldMessage(port))
            return
        }
        decisionHandler(.allow)
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        guard child?.isRunning == true else {
            tearDownWebView()
            return
        }
        showConsole()
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

    // Quit stops only a gateway the app started; an attached one keeps running.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let child = child as? Process, child.isRunning else { return .terminateNow }
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
            var counts: RunningCounts?
            var reply: Data?
            if let data = data, (response as? HTTPURLResponse)?.statusCode == 200,
               let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let sessions = object["sessions"] as? [String: Any] {
                running = sessions["running"] as? Int
                counts = runningCounts(data)
                reply = data
            }
            RunLoop.main.perform(inModes: [.default, .modalPanel]) {
                guard let self = self else { sender.reply(toApplicationShouldTerminate: false); return }
                guard child.isRunning else {
                    self.stopping = true
                    self.tearDownWebView()
                    sender.reply(toApplicationShouldTerminate: true)
                    return
                }
                // Ask only when something would stop. A turn that starts while the health request is in
                // flight is interrupted like any other, and the console offers to continue it on reopening.
                if quitNeedsConfirmation(reply) {
                    let alert = NSAlert()
                    alert.messageText = L10n.t("app.quit.ask", "Quit localmelo?")
                    alert.informativeText = quitMessage(counts, statusKnown: running != nil)
                    alert.addButton(withTitle: L10n.t("app.quit.stop", "Quit and Stop"))
                    alert.addButton(withTitle: L10n.t("app.quit.cancel", "Cancel"))
                    guard alert.runModal() == .alertFirstButtonReturn else {
                        self.quitPending = false
                        if !child.isRunning {
                            self.tearDownWebView()
                            sender.reply(toApplicationShouldTerminate: false)
                            DispatchQueue.main.async {
                                self.fail(L10n.t("app.fail.exitedWhileAsking", "The gateway exited while the quit confirmation was open."))
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

    // Shows the problem in the window, with Try Again, Open Log and Quit. Nothing here
    // ends the app: the child is stopped, and a new one starts only on Try Again.
    // `stopping` keeps callbacks still in flight from the failed child from going on.
    private func fail(_ text: String) {
        stopping = true
        timer?.invalidate()
        healthTimer?.invalidate()
        tearDownWebView()
        if let failed = child as? Process, failed.isRunning {
            kill(failed.processIdentifier, SIGTERM)
            // A gateway that ignored SIGTERM would keep the session store from the next start.
            DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
                if failed.isRunning { kill(failed.processIdentifier, SIGKILL) }
            }
        }
        child = nil
        menuBar.show(.unknown)
        pet.update(health: nil, stopped: true)
        pill.update(health: nil, stopped: true)
        pendingRoute = nil
        statusView.showFailure(failureNotice(reason: text, ready: ready))
        openWindow()
    }
}

let application = NSApplication.shared
let delegate = App()
application.delegate = delegate
application.run()
