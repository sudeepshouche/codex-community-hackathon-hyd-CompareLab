import "./globals.css";
import { TooltipProvider } from "@/components/ui/tooltip";

export const metadata = {
  title: "Compare Lab — Open Source A/B Testing",
  description: "Open source A/B testing: upload two versions, compare response curves, and log views and subscriptions to learn what holds.",
};

export const viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
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
