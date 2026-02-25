import json

with open('notebooks/cms_open_payments_analysis.ipynb', 'r') as f:
    nb = json.load(f)

new_cells = []
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "7️⃣ Anomaly Detection" in source:
            continue
        
        # Filter out lines containing "Anomaly" or "IQR" from markdown
        new_source = []
        for line in cell.get('source', []):
            if "Anomaly Detection" not in line and "Anomaly" not in line and "IQR" not in line:
                new_source.append(line)
        cell['source'] = new_source

    elif cell['cell_type'] == 'code':
        source = "".join(cell['source']).lower()
        
        # Skip entire cells dedicated to anomaly detection
        if "is_anomaly" in source or "top_anom" in source or "anomaly_detection.png" in source or "anomaly_by_nature" in source:
            continue
        
        # For the final plotting cell which does multiple things
        new_source = []
        skip = False
        for line in cell.get('source', []):
            if "# 5. anomaly amounts" in line.lower() or "anomaly" in line.lower():
                skip = True
                continue
            if skip and ("tight_layout" in line or "savefig" in line):
                skip = False
            
            if not skip:
                new_source.append(line)
        cell['source'] = new_source

    new_cells.append(cell)

nb['cells'] = new_cells

with open('notebooks/cms_open_payments_analysis.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Removed extra anomaly detection from notebook.")
