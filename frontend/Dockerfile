# syntax=docker/dockerfile:1

# ---- shared dependency layer ----
FROM docker.io/denoland/deno:2.8.1 AS deps
WORKDIR /app
COPY deno.json package.json ./
# Populates node_modules (nodeModulesDir=auto) for Vite/Svelte.
RUN deno install

# ---- dev server ----
# Source files are bind-mounted at runtime (see docker-compose.yml volumes),
# so rebuilding the image is only needed when deno.json / package.json change.
FROM deps AS dev
WORKDIR /app
EXPOSE 5173
CMD ["deno", "task", "dev"]

# ---- production build ----
FROM deps AS build
WORKDIR /app
COPY . .
# The bundle calls the API at its own origin, so no API base is baked in.
RUN deno task build
