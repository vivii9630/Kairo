import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Kairo AI",
  description: "Chat-native temporal RAG engine",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="h-screen overflow-hidden font-sans antialiased">
        {children}
      </body>
    </html>
  );
}
