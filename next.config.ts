import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // funny.kkoreng.com/shine-my-bald 아래에서 서비스 (funny-kkoreng-com 프로젝트가 이 경로를 rewrite)
  basePath: "/shine-my-bald",
  // 루트(shine-my-bald.vercel.app/, localhost:3000/)로 들어오면 앱으로 보낸다
  async redirects() {
    return [
      {
        source: "/",
        destination: "/shine-my-bald",
        basePath: false,
        permanent: false,
      },
    ];
  },
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
