import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // funny.kkoreng.com/shine-my-bald 아래에서 서비스 (funny-kkoreng-com 프로젝트가 이 경로를 rewrite)
  basePath: "/shine-my-bald",
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
