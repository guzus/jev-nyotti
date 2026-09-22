FROM node:24-bookworm-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY tsconfig*.json vite.config.ts ./
COPY server ./server
COPY web ./web
RUN npm run build && npm prune --omit=dev

FROM node:24-bookworm-slim
ENV NODE_ENV=production PORT=3000 DATA_DIR=/data
WORKDIR /app
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY package.json ./
COPY docs/api-example.json ./docs/api-example.json
RUN mkdir -p /data && chown node:node /data /app
USER node
EXPOSE 3000
CMD ["node", "dist/server/bootstrap.js"]
