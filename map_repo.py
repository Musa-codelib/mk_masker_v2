import os
from pathlib import Path

# Directories and files to skip completely
IGNORE_LIST = {
    '.DS_Store', 
    '__pycache__', 
    '.git', 
    '.idea', 
    '.vscode', 
    'node_modules',
    'Applications',      # Ignore the nested install target folder
    'Frameworks',  
    'venv',              # Ignore virtual environment folders
    '_CodeSignature'     # Ignore signature folders
}

def generate_tree(dir_path, prefix="", is_last=True):
    dir_path = Path(dir_path)
    lines = []
    
    # Don't map ignored directories
    if dir_path.name in IGNORE_LIST:
        return lines

    # Format the current item
    connector = "└── " if is_last else "├── "
    
    # Check if the folder is a macOS bundle (like .app or .framework)
    is_bundle = dir_path.suffix in {'.app', '.framework'}

    if prefix:
        if dir_path.is_dir():
            lines.append(f"{prefix}{connector}{dir_path.name}/")
        else:
            lines.append(f"{prefix}{connector}{dir_path.name}")
    else:
        lines.append(f"{dir_path.name}/")

    # If it's a directory AND not a bundle, map its children.
    # This keeps .app files visible in the tree without dumping their internal files.
    if dir_path.is_dir() and not is_bundle:
        # Update prefix for children
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        try:
            contents = [p for p in dir_path.iterdir() if p.name not in IGNORE_LIST]
            contents.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
            
            for i, child in enumerate(contents):
                is_child_last = (i == len(contents) - 1)
                lines.extend(generate_tree(child, new_prefix, is_child_last))
        except PermissionError:
            lines.append(f"{new_prefix}└── [Permission Denied]")
            
    return lines

def main():
    target_dir = Path(__file__).parent.resolve()
    print(f"Mapping directory: {target_dir}\n")
    
    tree_lines = generate_tree(target_dir)
    output_text = "\n".join(tree_lines)
    
    print(output_text)
    
    output_file = target_dir / "repository_map.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)
        
    print(f"\n✓ Map updated successfully! Saved to: {output_file}")

if __name__ == "__main__":
    main()