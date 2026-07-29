import { createContext, useContext, type ReactNode } from "react";

/** Whether the surrounding screen is the one on show. Outside a `Pane` — in a
 *  dialog, or a screen rendered on its own — everything is on show. */
const PaneActive = createContext(true);

export function usePaneActive(): boolean {
  return useContext(PaneActive);
}

/**
 * One screen of the deck, kept alive once it has been visited.
 *
 * A screen holds work that cost real time to produce: the lab's hand-built line
 * and its whole-line walk, history's aggregates at 60–110 s each, a live play
 * session. Unmounting on every nav click threw all of it away, so a visited
 * screen stays mounted and is hidden instead — and the ones never opened are
 * never mounted, so nothing queries on behalf of a screen the user has not asked
 * for. Only the visible pane binds the deck's document-level keys.
 */
export default function Pane({ active, children }: { active: boolean; children: ReactNode }) {
  return <PaneActive.Provider value={active}>
    <div className="screen-pane" hidden={!active}>{children}</div>
  </PaneActive.Provider>;
}
