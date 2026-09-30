from ml_services.map_renderer import render_folium_map_to_png
html = "<html><body><h1>Hello</h1></body></html>"
png = render_folium_map_to_png(html)
print("got bytes:", len(png) if png else None)