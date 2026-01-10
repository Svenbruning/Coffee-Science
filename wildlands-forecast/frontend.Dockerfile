# frontend.Dockerfile
FROM nginx:alpine

# eigen nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# statische site naar de default webroot
COPY web/ /usr/share/nginx/html/

