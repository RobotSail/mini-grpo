"""Simple HTTP server with file upload support."""
import os
import http.server
import cgi

UPLOAD_DIR = os.path.dirname(os.path.abspath(__file__))

class UploadHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        content_type = self.headers.get('Content-Type', '')
        if 'multipart/form-data' not in content_type:
            self.send_error(400, "Expected multipart/form-data")
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': content_type})

        if 'file' not in form:
            self.send_error(400, "No file field")
            return

        item = form['file']
        if item.filename:
            filepath = os.path.join(UPLOAD_DIR, os.path.basename(item.filename))
            with open(filepath, 'wb') as f:
                f.write(item.file.read())
            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"Saved: {filepath}\n".encode())
            print(f"Uploaded: {filepath}")
        else:
            self.send_error(400, "Empty filename")

    def do_GET(self):
        if self.path == '/upload':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b'''<html><body>
<h2>Upload File</h2>
<form method="POST" enctype="multipart/form-data" action="/">
<input type="file" name="file"><br><br>
<input type="submit" value="Upload">
</form></body></html>''')
        else:
            super().do_GET()

if __name__ == '__main__':
    os.chdir(UPLOAD_DIR)
    server = http.server.HTTPServer(('0.0.0.0', 8888), UploadHandler)
    print(f"Serving at :8888 with upload support")
    print(f"  Browse: http://localhost:8888/")
    print(f"  Upload: http://localhost:8888/upload")
    server.serve_forever()
