import "./globals.css";
import { TooltipProvider } from "@/components/ui/tooltip";

export const metadata = {
  title: "TRIBE Compare Lab",
  description: "Structured side-by-side video comparison using TRIBE-derived response curves.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <TooltipProvider>
          <div className="app-shell">{children}</div>
        </TooltipProvider>
      </body>
    </html>
  );
}
