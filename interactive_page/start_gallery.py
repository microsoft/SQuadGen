import http.server
import socketserver
import webbrowser
import os
import json
from pathlib import Path

# --- CONFIGURATION ---
PORT = 8000
HTML_FILENAME = "index.html"

def generate_html():
    # 1. Automatically find all subdirectories that contain .glb files
    data = []
    # Get all subdirectories and sort them (to solve your initial sorting question!)
    subdirs = sorted([d for d in Path('.').iterdir() if d.is_dir()])

    for folder in subdirs:
        # Check if the folder contains any of our target files
        found_files = []
        for i in range(4):
            filename = f"gen{i}.glb"
            if (folder / filename).exists():
                found_files.append(f"{folder.name}/{filename}")
        
        if found_files:
            data.extend(found_files)

    # 2. Create the HTML content with the dynamic list injected
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>3D Model Gallery</title>
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
    <style>
        body {{ font-family: sans-serif; background: #1a1a1a; color: white; margin: 0; display: flex; flex-direction: column; align-items: center; }}
        .header {{ padding: 20px; text-align: center; }}
        .grid {{ 
            display: grid; 
            grid-template-columns: repeat(4, 1fr); 
            grid-template-rows: repeat(4, 1fr); 
            gap: 15px; width: 95vw; height: 80vh; 
        }}
        .card {{ background: #2a2a2a; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; border: 1px solid #444; }}
        model-viewer {{ width: 100%; height: 100%; --poster-color: transparent; }}
        .label {{ font-size: 10px; padding: 5px; text-align: center; color: #aaa; background: #222; }}
        .nav {{ margin: 20px; display: flex; gap: 20px; align-items: center; }}
        button {{ padding: 10px 20px; cursor: pointer; background: #444; color: white; border: none; border-radius: 4px; }}
        button:hover {{ background: #666; }}
        button:disabled {{ opacity: 0.3; cursor: not-allowed; }}
    </style>
</head>
<body>
    <div class="header"><h1>Chart Distance Field Explorer</h1><p id="pageInfo"></p></div>
    <div id="gallery" class="grid"></div>
    <div class="nav">
        <button id="prevBtn" onclick="changePage(-1)">← Backward</button>
        <button id="nextBtn" onclick="changePage(1)">Forward →</button>
    </div>

    <script>
        const allModels = {json.dumps(data)};
        const itemsPerPage = 16;
        let currentPage = 0;

        function render() {{
            const container = document.getElementById('gallery');
            container.innerHTML = '';
            const start = currentPage * itemsPerPage;
            const pageItems = allModels.slice(start, start + itemsPerPage);

            pageItems.forEach(path => {{
                const div = document.createElement('div');
                div.className = 'card';
                div.innerHTML = `
                    <model-viewer src="${{path}}" auto-rotate camera-controls shadow-intensity="1"></model-viewer>
                    <div class="label">${{path}}</div>
                `;
                container.appendChild(div);
            }});

            document.getElementById('pageInfo').innerText = `Page ${{currentPage + 1}} of ${{Math.ceil(allModels.length / itemsPerPage)}}`;
            document.getElementById('prevBtn').disabled = currentPage === 0;
            document.getElementById('nextBtn').disabled = start + itemsPerPage >= allModels.length;
        }}

        function changePage(step) {{ currentPage += step; render(); window.scrollTo(0,0); }}
        render();
    </script>
</body>
</html>
"""
    with open(HTML_FILENAME, "w", encoding="utf-8") as f:
        f.write(html_template)

if __name__ == "__main__":
    print("Scanning folders and generating gallery...")
    generate_html()
    
    # Start Server
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"Server started at http://localhost:{PORT}")
        webbrowser.open(f"http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")