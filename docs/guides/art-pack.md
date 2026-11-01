# Art pack: replace every picture of Melo

One command checks a folder or a zip of Melo's art and replaces every picture of Melo in the repository with it: the desktop pet, the console's portrait, the faces, the tab icons, the app icon and the menu-bar logo. This guide is for whoever draws or sources the art and builds localmelo from a checkout.

![The poster of each of Melo's six states, with the state's name: startup, thinking, working, attention, idle_rest and idle_play.](../assets/img/art-pack-moves.webp)

## Quick start

- You need a checkout, `uv` and a pack: a folder or a `.zip` laid out as in [The pack](#the-pack).
- Pillow is not a dependency of localmelo. `--with pillow==13.0.0` fetches it for one run; the pin keeps the bytes the same, because another Pillow or libwebp may encode other bytes.
- The pack is only read: nothing in it is run, and a zip is checked before anything is written.

**1. Check.** It writes nothing to the checkout and exits 1 when the pack has problems:

```shell
uv run --with pillow==13.0.0 python scripts/art_pack.py check melo-pack.zip
```

The report says what the pack replaces, what is missing and what is used instead, the problems that stop `apply`, the warnings, and the files it ignored. A pack with two problems looks like this:

```console
$ uv run --with pillow==13.0.0 python scripts/art_pack.py check melo-pack
Art pack: melo-pack

Missing -> uses instead
  brand/app-icon.png: AppIcon.icns stays as it is
  brand/menubar.png: MenuBarLogo.png, MenuBarLogo@2x.png and the website's round logo stay as they are
  brand/emblem.png: docs/assets/melo.png (the README emblem) stays as it is

Problems (2)
  six states: the pack has no thinking
  durations: working: one positive duration (ms) per frame, as whole numbers

Warnings (0)
  none

Ignored (10)
  README.md
  *.apng (6 files)
  *.gif (6 files)
  preview.html
  transition-demo.js
  transition-player.js
  frames/ (16 files)
  integration/ (2 files)
  posters/ (6 files)
  sprites/ (6 files)

Result: the pack has problems; apply would refuse.
```

**2. Apply.** It checks again, then replaces every picture in one run and prints the changed files, a preview sheet and the next steps:

```shell
uv run --with pillow==13.0.0 python scripts/art_pack.py apply melo-pack.zip
```

- `-o DIR` leaves the preview sheet in `DIR`; without it a new temporary folder is made.
- `-r ROOT` writes under `ROOT` instead of this checkout, which is how to try a pack without touching it.
- The same pack always gives the same bytes.

**3. Look.** Open the preview sheet it prints: every poster, the stills, the faces at the sizes they are shown, the tab icons and the brand pictures, on a light page and a dark one.

**4. Test, commit, rebuild.** Run the picture tests, commit what `git status` lists, and rebuild the app:

```shell
uv run --with pillow==13.0.0 pytest tests/test_moves_assets.py tests/cli/test_mark.py -q
uv run python scripts/build_macos_app.py --install
```

The screenshots in `docs/assets/img` show the console and the app with the old art; retake them when the pictures change.

## The pack

A pack is a folder, or a `.zip` whose root is that folder or holds it as its one top folder. Only PNG and JSON files are read.

```text
melo-pack/
  manifest.json                  the states, frames, timing, poster frames and the canvas
  frames/<state>/frame_NN.png    the six states, 1 to 16 frames each
  stills/done.png                optional: a run that ended well
  stills/error.png               optional: a failure
  stills/offline.png             optional: nothing known, or no connection
  heads/head-smile.png           optional: the head for small sizes
  heads/head-neutral.png         optional: the calm head for small sizes
  brand/app-icon.png             optional: the macOS app icon
  brand/menubar.png              optional: the menu-bar logo and the website's round logo
  brand/emblem.png               optional: the README emblem
```

- The six states are `startup`, `thinking`, `working`, `attention`, `idle_rest` and `idle_play`: exactly these, each with at least one frame.
- Frames are named `frame_NN.png`, counted without a gap from `frame_00` or from `frame_01`, in play order. At most 16 frames a state (the sheet is 4 x 4).
- The stills and the heads are also accepted at the pack's root. When both places hold one, the folder's copy is used and the report warns.
- Every other file is listed as ignored, never an error. A pack may keep its previews, sprites and players beside the art.
- A zip is checked before it is read: see [Zip and folder safety](#zip-and-folder-safety).

The manifest names the frames and the timing of each state, and the canvas. These are the fields that are read; others may stay:

```json
{
  "canvas": {"width": 512, "height": 512, "anchor": {"x": 256, "y": 488}},
  "states": {
    "startup": {
      "frames": ["frames/startup/frame_01.png", "frames/startup/frame_02.png"],
      "frame_durations_ms": [450, 200],
      "playback": "once",
      "poster_frame_index": 1
    }
  }
}
```

- `frame_durations_ms` has one positive whole number of milliseconds per frame.
- `playback` is `"once"` or `"loop"`; `loop_count` of 1 also means once, and 0 means loop.
- `poster_frame_index` counts from 0 and defaults to the first frame.
- A manifest that also lists other states is accepted with a warning; only the six are used.

## The six states

| State | Melo | The console and the pet play it when | Plays | The current pack |
|---|---|---|---|---|
| `startup` | Wakes up, shakes and sits | The page or the app has just opened | Once, then holds its last frame | 16 frames, 4.0 s |
| `thinking` | Tilts its head to one side, then the other | The model is answering and no tool runs | Loops | 16 frames, 5.6 s |
| `working` | Types at its laptop | A chat, task or run is working | Loops | 16 frames, 1.3 s |
| `attention` | Raises a paw and taps | Something waits for your approval; it shows at once | Loops | 16 frames, 1.6 s |
| `idle_rest` | Lies down, eyes closed, breathing | Nothing runs | Loops | 16 frames, 3.2 s |
| `idle_play` | Plays with its toy | After about 20 seconds of rest, at most every 5 minutes | Loops; one pass is played | 16 frames, 6.0 s |

- A frame lasts from 40 ms to 1.6 s in the current pack; the frames of one state may last different times.
- The poster frame is the one picture that stands for the state when nothing may move. `startup` uses its last frame, the sitting dog; the others use their first.
- Each state keeps one pose, which the code fixes: `startup`, `thinking` and `attention` sit, `working` is at the desk, `idle_rest` lies and `idle_play` stands.
- A change between two states cross-fades: 160 ms within a pose, 220 ms between poses and 120 ms into `attention`, which interrupts anything.
- `startup` is meant to play once and `working`, `thinking`, `attention` and `idle_rest` to loop; the report warns when a pack does otherwise.

See [The macOS app](macos-app.md#melo-on-your-desktop) and [The web console](web-console.md#the-panel-beside-a-chat) for how the states look in use.

## Stills and heads

The stills and the heads are optional. Each is one transparent picture on the pack's canvas, the paws on the anchor line.

| File | Shows | Where it appears |
|---|---|---|
| `stills/done.png` | Eyes closed and happy, tail up | The panel and the desktop pet for a few seconds after a run ends well; the face on the newest reply |
| `stills/error.png` | Ears back, head low | A run that just failed or a model server that cannot be reached; the face on that reply |
| `stills/offline.png` | Curled up | Nothing known yet: connecting, or localmelo does not answer |
| `heads/head-smile.png` | A front-facing, smiling head | The reply avatar, the Main chat and the phone's panel chip |
| `heads/head-neutral.png` | A calm head, ears up | The sidebar and the sign-in screen, the browser tab at 16, 32 and 64 px, and the terminal welcome's mark |

- The faces that the activity pill, the approval bar and the replies show are cut from the posters and stills by fixed boxes; a head replaces the cut where it is given.
- A head is the bust fitted in the head box (see [The canvas](#the-canvas)): 464 px square of a 512 px canvas, its bottom on the anchor line.
- The tab icon is cut closer from the calm head, with the neck left out, so it still reads at 16 px.
- The terminal welcome's mark is computed from the calm head, so `support/cli/mark.py` changes with it.

## Brand pictures

The brand pictures are optional and are not on the canvas. They are PNG files of any size up to 4096 px.

| File | Must be | Becomes |
|---|---|---|
| `brand/app-icon.png` | Square, 1024 px or more (512 px is a warning) | `macos/assets/AppIcon.icns`, every size |
| `brand/menubar.png` | Square | `MenuBarLogo.png` (18 px), `MenuBarLogo@2x.png` (36 px) and the website's round logo, 64 px |
| `brand/emblem.png` | Under 2 MB | `docs/assets/melo.png`, the README emblem; also the app icon when there is no `brand/app-icon.png` |

- Draw the app icon as a full square with a flat paper-coloured background: the build lays it on a rounded tile of the macOS icon grid and fades its edges into the tile's colour.
- A picture with transparent pixels is laid on white first, and the report warns.
- Colours are read as sRGB; a picture with another colour profile warns.

## The canvas

![The canvas of 512 px with the anchor line, the head box and the two tab-icon boxes.](../assets/img/art-pack-canvas.webp)

- One square canvas for every frame, still and head: 512 x 512 px in the current pack. Anything from 256 to 4096 px a side works; a smaller canvas warns, because the pictures are made at 256 px and would be scaled up.
- `canvas.anchor` is the point the paws rest on, inside the canvas: (256, 488) in the current pack. The whole canvas is scaled, never one frame cut on its own, so the paws stay on the line.
- The report warns when the lowest drawn row of a picture lies more than 2% of the canvas from the anchor line. The art's own shadow may sit a little under the paws.
- The background is transparent: RGBA PNG, with the four corners clear and something drawn. Edges are antialiased to transparent, with no light or dark fringe from a coloured background.
- The sizes in the figure are for a 512 px canvas; a larger or smaller one scales them.

## When something is missing

| Missing | Used instead |
|---|---|
| `stills/done.png`, `error.png`, `offline.png` | The matching poster is a copy of the `startup`, `thinking` or `idle_rest` poster; the done and error faces are cut from the `startup` and `thinking` posters |
| `heads/head-smile.png` | The reply face is cut from the `startup` poster |
| `heads/head-neutral.png` | The brand face is the reply face, and the tab icons are cut from the `startup` poster |
| `brand/app-icon.png` | The app icon is drawn from `brand/emblem.png`, else `AppIcon.icns` stays as it is |
| `brand/menubar.png` | `MenuBarLogo.png`, `MenuBarLogo@2x.png` and the website's round logo stay |
| `brand/emblem.png` | `docs/assets/melo.png` stays |

- The moves folder always holds all nine posters, so the console and the app never find a hole.
- The report lists each of these under **Missing -> uses instead** before anything is written.

## Size and reduced motion

- The moves in `support/gateway/static/moves` weigh about 1.7 MB now and may not pass 2 MB; `apply` refuses a pack that goes over (the rule is `size budget`).
- Each frame is scaled to 256 px and saved as lossy WebP with a lossless alpha. Fewer frames or flatter shading make a pack lighter.
- With reduced motion each state is its poster frame and a change is an 80 ms fade, so each poster must tell its state without moving.
- While the page is hidden, or Melo is covered, nothing is drawn.

## The character

Melo is an Akita Inu. This is guidance for the art, not a check: nothing tests it, and the art is yours.

- A white or cream mask between the eyes and over the brows, and a white muzzle and cheeks.
- Orange only on top of the head, the ears, the back and the outer legs.
- A dark outline and orange fur keep Melo readable on the console's light and dark pages.

## What the check tests

Each finding starts with the rule in plain words. A problem stops `apply`; a warning does not.

| Problem | What it tests |
|---|---|
| `zip safety`, `folder safety` | The archive or folder is safe to read (see below) |
| `manifest` | `manifest.json` is one readable JSON object |
| `square canvas`, `sane size` | The canvas is square; no side is over 4096 px and no picture over 50 MB |
| `anchor` | The canvas has an anchor with numbers `x` and `y`, inside it |
| `six states` | All six states are there, each an object |
| `frame count`, `frame names`, `frame files` | 1 to 16 frames a state, named in order, each a file of the pack |
| `durations`, `playback`, `poster frame` | One positive whole duration per frame; `once` or `loop`; the poster is one of the frames |
| `readable PNG`, `on the canvas`, `RGBA` | Each picture is a real PNG with an alpha channel, on the canvas |
| `transparent corners` | The four corners are clear and something is drawn |
| `app icon`, `menu-bar logo` | The brand pictures are square, the app icon at least 512 px |
| `size budget` | The moves stay under 2 MB |

| Warning | What it tests |
|---|---|
| `canvas size` | The canvas is under 256 px, so the pictures are scaled up |
| `six states`, `playback` | The manifest lists other states; a state plays differently from how it is used |
| `paws on the anchor line` | A picture rests more than 2% of the canvas from the line |
| `head box` | A head draws outside its box and is cut off |
| `stills`, `heads` | A file is given at the root and in its folder |
| `colour profile`, `emblem`, `app icon`, `menu-bar logo` | A profile that is not sRGB, a heavy emblem, an icon under 1024 px or with transparent pixels, a logo under 64 px |
| `terminal mark`, `encoder` | The mark cannot be made; Pillow is not the pinned version, so the bytes may differ from the committed ones |

## What apply writes

- The moves folder, `support/gateway/static/moves`, is replaced whole: the sheets, the posters, the stills, the faces, the tab icons and `manifest.json`. A file that is not in the new set is removed.
- `support/cli/mark.py`, the terminal welcome's mark, is computed from the new calm face.
- With brand pictures: `macos/assets/AppIcon.icns`, `MenuBarLogo.png`, `MenuBarLogo@2x.png`, `docs/assets/melo.png` and `docs/assets/icon_small.png`.
- The two pictures of this guide, `docs/assets/img/art-pack-moves.webp` and `art-pack-canvas.webp`, are drawn again from the new posters. `art_pack.py figures` redraws them alone.

Nothing from the pack is copied except the brand pictures it names. The pack itself stays outside the repository.

## Zip and folder safety

A pack comes from outside, so it is untrusted data. Only its names, sizes and PNG and JSON bytes are read.

- A zip is checked whole before one byte is written; a refusal writes nothing.
- It is refused for an absolute path, `..`, a backslash or a drive letter in a name; a symbolic link or any entry that is not a regular file or a folder; an encrypted entry; or two names that differ only in case.
- It is refused above 2000 entries, 50 MB for one file or 300 MB in all, counted again as the bytes are read.
- Only PNG and JSON are extracted, into a new temporary folder that is removed after. The `__MACOSX` folders, `.DS_Store` and `._` files that a Mac adds are ignored.
- A folder is read in place, with symbolic links refused and the same caps.
