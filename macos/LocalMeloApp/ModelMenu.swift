import Foundation

/// melo_serve's chat models for the menu bar's Model submenu: `GET /api/serve`, worded by the gateway in the
/// app's language, so the rows read as the console's and the terminal's. The submenu shows only while melo_serve
/// serves the chats (`model.backend` of `GET /api/health`).
struct ServePick: Equatable {
    let id: String
    let name: String
    let words: String
    let current: Bool
    let loadable: Bool
}

struct ServeList: Equatable {
    let picks: [ServePick]
    let foot: String

    var current: ServePick? { picks.first { $0.current } }
}

/// A switch of the served model, from `switch` of `GET /api/health`: `title` and `detail` are its two lines,
/// for one that failed what it did not reach and why.
struct ServeSwitch: Equatable {
    let id: Int
    let pick: String
    let state: String
    let title: String
    let detail: String

    /// Waiting for chats, or loading: the picks wait for it.
    var active: Bool { state == "waiting" || state == "loading" }
    var failed: Bool { state == "failed" }
}

private func json(_ data: Data) -> [String: Any]? {
    try? JSONSerialization.jsonObject(with: data) as? [String: Any]
}

/// Whether melo_serve serves the chats, so their model switches from the menu.
func servesModels(_ health: Data) -> Bool {
    (json(health)?["model"] as? [String: Any])?["backend"] as? String == "melo_serve"
}

/// The served model's name in `GET /api/health`, "" when none: a change reads the picks again.
func servedModel(_ health: Data) -> String {
    (json(health)?["model"] as? [String: Any])?["name"] as? String ?? ""
}

/// The last switch the health names, nil before the first one or for a damaged one.
func serveSwitch(_ health: Data) -> ServeSwitch? {
    guard let entry = json(health)?["switch"] as? [String: Any] else { return nil }
    return switchEntry(entry)
}

/// A switch as `POST /api/serve/switch` answers it (`{"switch": ...}`).
func startedSwitch(_ data: Data) -> ServeSwitch? {
    guard let entry = json(data)?["switch"] as? [String: Any] else { return nil }
    return switchEntry(entry)
}

private func switchEntry(_ entry: [String: Any]) -> ServeSwitch? {
    guard let id = entry["id"] as? Int, let pick = entry["pick"] as? String,
          let state = entry["state"] as? String else { return nil }
    return ServeSwitch(id: id, pick: pick, state: state, title: plainLine(entry["title"] as? String, limit: 120),
                       detail: plainLine(entry["detail"] as? String, limit: 120))
}

/// The picks of `GET /api/serve`; a row with no id or name is left out, and nil for a reply that lists none.
func serveList(_ data: Data) -> ServeList? {
    guard let body = json(data), let rows = body["picks"] as? [[String: Any]] else { return nil }
    let picks = rows.prefix(64).compactMap { row -> ServePick? in
        guard let id = row["id"] as? String, !id.isEmpty, let name = row["name"] as? String, !name.isEmpty else { return nil }
        return ServePick(id: id, name: plainLine(name, limit: 80), words: plainLine(row["words"] as? String, limit: 120),
                         current: row["current"] as? Bool == true, loadable: row["loadable"] as? Bool == true)
    }
    return ServeList(picks: picks, foot: plainLine(body["foot"] as? String, limit: 120))
}

/// One line of the Model submenu.
enum ModelMenuRow: Equatable {
    /// The switch under way: what it switches to and what it waits for. Not chosen.
    case status(title: String, subtitle: String)
    /// Ends the wait of a switch that has not started loading.
    case cancel
    case separator
    case pick(id: String, title: String, subtitle: String, checked: Bool, enabled: Bool)
    /// "Every chat on this Mac follows the switch." Not chosen.
    case foot(String)
}

/// The submenu: while a switch runs, its line and Cancel Switch first and every pick dimmed; after one that failed,
/// why, until the next; a pick that cannot load is always dimmed, its reason its subtitle; the current one checked.
func modelMenuRows(_ list: ServeList, switching: ServeSwitch?) -> [ModelMenuRow] {
    var rows: [ModelMenuRow] = []
    if let running = switching, running.active {
        rows.append(.status(title: running.title, subtitle: running.detail))
        if running.state == "waiting" { rows.append(.cancel) }
        rows.append(.separator)
    } else if let ended = switching, ended.failed {
        rows += [.status(title: ended.title, subtitle: ended.detail), .separator]
    }
    let busy = switching?.active == true
    for pick in list.picks {
        rows.append(.pick(id: pick.id, title: pick.name, subtitle: pick.words, checked: pick.current,
                          enabled: pick.loadable && !busy))
    }
    if !list.foot.isEmpty { rows += [.separator, .foot(list.foot)] }
    return rows
}

/// Whether to draw *rows*: a poll with nothing new leaves the submenu as it is, and an open one is not rebuilt under
/// the pointer (the first poll after it closes draws it).
func modelMenuNeedsDrawing(_ rows: [ModelMenuRow], shown: [ModelMenuRow], open: Bool) -> Bool {
    rows != shown && !open
}

/// The Model item's subtitle: the switch under way, else the current model's name.
func modelMenuSubtitle(_ list: ServeList?, switching: ServeSwitch?) -> String {
    if let running = switching, running.active { return running.title }
    return list?.current?.name ?? ""
}
