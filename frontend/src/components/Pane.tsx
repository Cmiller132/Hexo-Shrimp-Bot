import { createContext, useContext, type ReactNode } from "react";

/** Whether the surrounding `Pane` is visible; consumers outside a `Pane` are active. */
const PaneActive = createContext(true);

export function usePaneActive(): boolean {
  return useContext(PaneActive);
}

/**
 * Keeps a visited screen mounted while exposing whether it is visible.
 * Local state survives navigation, and only the visible pane binds document-level keys.
 */
export default function Pane({ active, children }: { active: boolean; children: ReactNode }) {
  return <PaneActive.Provider value={active}>
    <div className="screen-pane" hidden={!active}>{children}</div>
  </PaneActive.Provider>;
}
