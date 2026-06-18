import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Header } from "@/components/Header";
import { ThemeProvider } from "@/components/ThemeProvider";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({
	variable: "--font-geist-mono",
	subsets: ["latin"],
});

export const metadata: Metadata = {
	title: "RL Benchmark Dashboard",
	description: "Reproducible reinforcement-learning benchmark results",
};

/**
 * Root layout: applies fonts, the theme provider and the sticky header shell.
 *
 * @param children - The active page content
 */
export default function RootLayout({
	children,
}: {
	children: React.ReactNode;
}) {
	return (
		<html lang="fr" suppressHydrationWarning>
			<body
				className={`${geistSans.variable} ${geistMono.variable} antialiased`}
			>
				<ThemeProvider>
					<Header />
					<main className="mx-auto max-w-[1200px] px-4 sm:px-6 py-10 animate-fade-in">
						{children}
					</main>
				</ThemeProvider>
			</body>
		</html>
	);
}
