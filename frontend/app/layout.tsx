import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Shrimp Control Deck",
  description:
    "A dense local console for playing Hexo, tracking training runs, reviewing games, and inspecting model internals.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
