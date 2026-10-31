import AppKit

/// The words of the app's menu items, by item. An item names its words (`item.word(.open)`), so a menu can write its
/// titles again when the console's language changes (`retitle`), without the app starting over.
enum MenuWord: String {
    case open, settings, signIn, quit, edit, undo, redo, cut, copy, paste, selectAll
    case waiting, showPet, hidePet, desktop, desktopPill, desktopPet, desktopNothing

    /// The item's title in the language now in use. *attached* picks the Quit that leaves a running gateway alone.
    func title(attached: Bool = false) -> String {
        switch self {
        case .open: return L10n.t("app.menu.open", "Open localmelo")
        case .settings: return L10n.t("app.menu.settings", "Settings…")
        case .signIn: return L10n.t("app.menu.signIn", "Sign In Again")
        case .quit: return quitTitle(attached: attached)
        case .edit: return L10n.t("app.edit.menu", "Edit")
        case .undo: return L10n.t("app.edit.undo", "Undo")
        case .redo: return L10n.t("app.edit.redo", "Redo")
        case .cut: return L10n.t("app.edit.cut", "Cut")
        case .copy: return L10n.t("app.edit.copy", "Copy")
        case .paste: return L10n.t("app.edit.paste", "Paste")
        case .selectAll: return L10n.t("app.edit.selectAll", "Select All")
        case .waiting: return L10n.t("app.menu.waiting", "Waiting for your approval")
        case .showPet: return L10n.t("app.menu.showPet", "Show Melo")
        case .hidePet: return L10n.t("app.pet.hide", "Hide Melo")
        case .desktop: return desktopMenuTitle
        case .desktopPill: return DesktopMode.pill.title
        case .desktopPet: return DesktopMode.pet.title
        case .desktopNothing: return DesktopMode.nothing.title
        }
    }
}

extension DesktopMode {
    /// The menu item that offers this choice.
    var menuWord: MenuWord {
        switch self {
        case .pill: return .desktopPill
        case .pet: return .desktopPet
        case .nothing: return .desktopNothing
        }
    }
}

extension NSMenuItem {
    /// Names the item's words and gives it its title in the language now in use.
    @discardableResult
    func word(_ word: MenuWord, attached: Bool = false) -> NSMenuItem {
        identifier = NSUserInterfaceItemIdentifier(word.rawValue)
        title = word.title(attached: attached)
        submenu?.title = title
        return self
    }
}

/// Writes the title of every item of *menu* and its submenus that names its words, in the language now in use.
/// An item without words (a line of the status, a separator) is left as it is.
func retitle(_ menu: NSMenu, attached: Bool = false) {
    for item in menu.items {
        if let raw = item.identifier?.rawValue, let word = MenuWord(rawValue: raw) {
            item.title = word.title(attached: attached)
            item.submenu?.title = item.title
        }
        if let submenu = item.submenu { retitle(submenu, attached: attached) }
    }
}
