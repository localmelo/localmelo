import Foundation

/// The app's words in the console's language: English or Simplified Chinese.
///
/// English lives in the code, beside each key: `L10n.t("app.menu.open", "Open localmelo")`. Another
/// language comes from the bundle's `<language>.lproj/Localizable.strings`, which the build writes from
/// the console's catalog (the `app.*` keys of `support/gateway/static/i18n/<language>.json`), so the app,
/// the console page and the gateway read one catalog. `{name}` marks a parameter, as in the catalog.
enum L10n {
    static let languages = ["en", "zh-Hans"]
    private(set) static var language = "en"
    private static var strings: Bundle?

    /// Use *language*'s strings from *bundle*; English, or a language the bundle lacks, needs none.
    static func use(_ language: String, bundle: Bundle = .main) {
        let wanted = languages.contains(language) ? language : "en"
        let path = wanted == "en" ? nil : bundle.path(forResource: wanted, ofType: "lproj")
        strings = path.flatMap { Bundle(path: $0) }
        self.language = strings == nil ? "en" : wanted
    }

    /// The words of *key*: *english* with its parameters filled in, or the chosen language's.
    static func t(_ key: String, _ english: String, _ params: [String: String] = [:]) -> String {
        fill(strings?.localizedString(forKey: key, value: english, table: nil) ?? english, params)
    }

    /// A count's words: English has one and other forms (`{n}` is the count), Chinese one form.
    static func plural(_ key: String, _ n: Int, one: String, other: String,
                       _ params: [String: String] = [:]) -> String {
        t(key, n == 1 ? one : other, params.merging(["n": String(n)]) { mine, _ in mine })
    }

    /// *text* with each `{name}` that *params* has replaced, in one pass: a value is never read again.
    static func fill(_ text: String, _ params: [String: String]) -> String {
        guard !params.isEmpty else { return text }
        var result = "", rest = Substring(text)
        while let open = rest.firstIndex(of: "{") {
            result += rest[..<open]
            let after = rest[rest.index(after: open)...]
            if let close = after.firstIndex(of: "}"), let value = params[String(after[..<close])] {
                result += value
                rest = after[after.index(after: close)...]
            } else {
                result += "{"
                rest = after
            }
        }
        return result + rest
    }
}

/// The language to show: *setting* when it names one, else the first preferred language the app has
/// (any `zh` but Traditional Chinese is Simplified Chinese), else English. The console page and the
/// gateway pick the same way.
func pickLanguage(setting: String, preferred: [String]) -> String {
    if L10n.languages.contains(setting) { return setting }
    for tag in preferred {
        let parts = tag.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: "_", with: "-")
            .lowercased().split(separator: "-").map(String.init)
        guard let first = parts.first else { continue }
        if first == "en" { return "en" }
        guard first == "zh" else { continue }
        let rest = Set(parts.dropFirst())
        if rest.contains("hans") || rest.isDisjoint(with: ["hant", "tw", "hk", "mo"]) { return "zh-Hans" }
    }
    return "en"
}

/// `[console] language` of the config file's text: `auto` when the file, the table or the key is missing.
func consoleLanguageSetting(_ config: String?) -> String {
    var table = ""
    for raw in (config ?? "").components(separatedBy: .newlines) {
        let line = raw.split(separator: "#", maxSplits: 1, omittingEmptySubsequences: false).first
            .map { $0.trimmingCharacters(in: .whitespaces) } ?? ""
        if line.hasPrefix("[") {
            table = line.trimmingCharacters(in: CharacterSet(charactersIn: "[] "))
            continue
        }
        let pair = line.split(separator: "=", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
        guard table == "console", pair.count == 2, pair[0] == "language" else { continue }
        return pair[1].trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
    }
    return "auto"
}

/// The script the console page runs first: the Mac's preferred languages, for its own choice.
func languagesScript(_ preferred: [String]) -> String {
    let list = (try? JSONSerialization.data(withJSONObject: preferred)).map { String(decoding: $0, as: UTF8.self) }
    return "window.localmeloLanguages = \(list ?? "[]");"
}
