import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
	output: "export",
	images: { unoptimized: true },
	basePath: process.env.NEXT_PUBLIC_BASE_PATH ?? "",
	turbopack: { root: path.resolve() },
};

export default nextConfig;
