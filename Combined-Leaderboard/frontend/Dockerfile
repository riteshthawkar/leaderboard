FROM node:22-bookworm-slim AS build

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . ./

ARG VITE_API_BASE_URL
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
ARG VITE_PRIVACY_POLICY_URL
ENV VITE_PRIVACY_POLICY_URL=${VITE_PRIVACY_POLICY_URL}
RUN npm run build

FROM nginx:1.27-alpine

ARG VITE_API_BASE_URL
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY security-headers.conf /etc/nginx/snippets/ms-vista-security-headers.conf
COPY --from=build /app/dist /usr/share/nginx/html
RUN sed -i "s|__API_ORIGIN__|${VITE_API_BASE_URL}|g" /etc/nginx/snippets/ms-vista-security-headers.conf \
    && grep -q "connect-src 'self' ${VITE_API_BASE_URL}" /etc/nginx/snippets/ms-vista-security-headers.conf

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8080/ >/dev/null || exit 1
