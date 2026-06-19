"use client";

import { Activity, Github } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "./ThemeToggle";

const NAV = [
	{ href: "/", label: "Runs" },
	{ href: "/compare", label: "Comparer" },
];

/**
 * Sticky top header with the app title, primary navigation and the theme toggle.
 */
export function Header() {
	const pathname = usePathname();

	return (
		<header className="sticky top-0 z-50 border-b border-[var(--border-subtle)] bg-[var(--header-bg)] backdrop-blur-md">
			<div className="mx-auto flex h-14 max-w-[1200px] items-center justify-between px-4 sm:px-6">
				<Link href="/" className="flex items-center gap-2.5">
					<Activity className="h-5 w-5 text-[var(--accent)]" />
					<span className="font-semibold">RL Benchmarks</span>
					<span className="hidden text-sm text-[var(--text-muted)] sm:block">
						Dashboard
					</span>
				</Link>

				<div className="flex items-center gap-1">
					{NAV.map((item) => {
						const active =
							item.href === "/"
								? pathname === "/"
								: pathname.startsWith(item.href);
						return (
							<Link
								key={item.href}
								href={item.href}
								className={`rounded-lg px-3 py-1.5 text-sm transition-all ${
									active
										? "bg-[var(--bg-muted)] font-medium text-[var(--text-primary)]"
										: "text-[var(--text-secondary)] hover:bg-[var(--bg-card)] hover:text-[var(--text-primary)]"
								}`}
							>
								{item.label}
							</Link>
						);
					})}
					<ThemeToggle />
					<a
						href="https://github.com"
						target="_blank"
						rel="noopener noreferrer"
						className="p-2 text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)]"
						aria-label="GitHub"
					>
						<Github className="h-4 w-4" />
					</a>
				</div>
			</div>
		</header>
	);
}
