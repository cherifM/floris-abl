#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert markdown reports to LaTeX and PDF format.

This script converts the blockage model validation reports from markdown to
LaTeX format, then compiles the LaTeX files to produce PDF documents.

MIT License

Copyright (c) 2025 Cherif Mihoubi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import shutil


def check_dependencies():
    """
    Check if required dependencies for conversion are installed.
    
    Returns:
        bool: True if all dependencies are installed, False otherwise
    """
    dependencies = {
        "pandoc": "Pandoc is required for Markdown to LaTeX conversion. Install with 'sudo apt-get install pandoc'",
        "pdflatex": "pdfLaTeX is required for LaTeX to PDF conversion. Install with 'sudo apt-get install texlive-latex-base texlive-latex-recommended texlive-latex-extra'"
    }
    
    missing = []
    
    for dep, message in dependencies.items():
        try:
            subprocess.run(["which", dep], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            missing.append((dep, message))
    
    if missing:
        print("Missing dependencies:")
        for dep, message in missing:
            print(f"  - {dep}: {message}")
        return False
    
    return True


def md_to_latex(md_file, output_dir, template=None):
    """
    Convert a markdown file to LaTeX format.
    
    Args:
        md_file (Path): Path to markdown file
        output_dir (Path): Directory to save the LaTeX file
        template (Path, optional): Path to LaTeX template file
        
    Returns:
        Path: Path to the generated LaTeX file, or None if conversion failed
    """
    print(f"Converting {md_file.name} to LaTeX...")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Output LaTeX file path
    tex_file = output_dir / f"{md_file.stem}.tex"
    
    # Build the pandoc command
    cmd = [
        "pandoc",
        str(md_file),
        "-f", "markdown",
        "-t", "latex",
        "-o", str(tex_file),
        "--standalone",
        "--toc",
        "--toc-depth=3",
        "--number-sections",
    ]
    
    # Add template if provided
    if template and Path(template).exists():
        cmd.extend(["--template", str(template)])
    
    # Run the conversion
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Successfully converted to {tex_file}")
        return tex_file
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr.decode('utf-8')}")
        return None


def latex_to_pdf(tex_file, output_dir):
    """
    Convert a LaTeX file to PDF.
    
    Args:
        tex_file (Path): Path to LaTeX file
        output_dir (Path): Directory to save the PDF file
        
    Returns:
        Path: Path to the generated PDF file, or None if conversion failed
    """
    print(f"Converting {tex_file.name} to PDF...")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Switch to the directory containing the LaTeX file
    original_dir = os.getcwd()
    os.chdir(tex_file.parent)
    
    # Run pdflatex (twice to resolve references)
    pdf_file = tex_file.with_suffix('.pdf')
    
    try:
        # First run
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_file.name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Second run to resolve references
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_file.name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Copy the PDF to the output directory if different
        if str(output_dir.resolve()) != str(tex_file.parent.resolve()):
            pdf_file_in_output = output_dir / pdf_file.name
            shutil.copy2(pdf_file, pdf_file_in_output)
            pdf_file = pdf_file_in_output
        
        print(f"✅ Successfully generated {pdf_file}")
        
        # Clean up temporary LaTeX files
        for ext in ['.aux', '.log', '.toc', '.out']:
            tmp_file = tex_file.with_suffix(ext)
            if tmp_file.exists():
                tmp_file.unlink()
                
        # Return to original directory
        os.chdir(original_dir)
        
        return pdf_file
    
    except subprocess.CalledProcessError as e:
        print(f"❌ PDF generation failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr.decode('utf-8')}")
        
        # Return to original directory
        os.chdir(original_dir)
        
        return None


def copy_images(md_file, output_dir):
    """
    Copy images referenced in the markdown file to the output directory.
    
    Args:
        md_file (Path): Path to markdown file
        output_dir (Path): Directory to save the images
        
    Returns:
        list: List of copied image files
    """
    # Get the content of the markdown file
    md_content = md_file.read_text()
    
    # Create images directory in output_dir
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True, parents=True)
    
    # Find image references in markdown (e.g., ![alt](path/to/image.png))
    import re
    image_pattern = r'!\[.*?\]\((.*?)\)'
    image_paths = re.findall(image_pattern, md_content)
    
    copied_images = []
    
    for img_path in image_paths:
        # Get absolute path to the image
        if os.path.isabs(img_path):
            src_img = Path(img_path)
        else:
            src_img = md_file.parent / img_path
        
        if src_img.exists():
            # Copy image to output directory
            dst_img = images_dir / src_img.name
            shutil.copy2(src_img, dst_img)
            copied_images.append(dst_img)
            
            # Update image path in LaTeX file (relative to the LaTeX file)
            # Note: This is a crude placeholder; proper handling would require modifying the LaTeX file
            # which is beyond the scope of this script
        else:
            print(f"Warning: Image file not found: {src_img}")
    
    return copied_images


def process_report(md_file, output_dir, template=None):
    """
    Process a single markdown report: convert to LaTeX and then to PDF.
    
    Args:
        md_file (Path): Path to markdown file
        output_dir (Path): Directory to save output files
        template (Path, optional): Path to LaTeX template file
        
    Returns:
        tuple: (tex_file, pdf_file) paths or None if conversion failed
    """
    # Copy images
    copy_images(md_file, output_dir)
    
    # Convert markdown to LaTeX
    tex_file = md_to_latex(md_file, output_dir, template)
    if not tex_file:
        return None, None
    
    # Convert LaTeX to PDF
    pdf_file = latex_to_pdf(tex_file, output_dir)
    
    return tex_file, pdf_file


def main():
    """Main entry point for the report conversion script."""
    parser = argparse.ArgumentParser(description="Convert markdown reports to LaTeX and PDF")
    parser.add_argument("--output", "-o", type=str, default="report_output",
                       help="Output directory for LaTeX and PDF files")
    parser.add_argument("--template", "-t", type=str, 
                       help="Path to LaTeX template file")
    parser.add_argument("--report", "-r", type=str,
                       help="Process specific report instead of all")
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        print("Please install missing dependencies and try again.")
        return 1
    
    # Get the directory containing this script
    script_dir = Path(__file__).resolve().parent
    
    # Create output directory
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get reports to process
    if args.report:
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = script_dir / report_path
        
        if not report_path.exists():
            print(f"Error: Report {args.report} not found.")
            return 1
        
        reports = [report_path]
    else:
        # Process all markdown reports in the directory
        reports = list(script_dir.glob("*.md"))
    
    if not reports:
        print("No markdown reports found.")
        return 1
    
    # Process each report
    successful = 0
    
    for report in reports:
        print(f"\nProcessing report: {report.name}")
        tex_file, pdf_file = process_report(report, output_dir, args.template)
        
        if pdf_file:
            successful += 1
    
    # Print summary
    print("\n" + "="*40)
    print(f"Conversion Summary: {successful}/{len(reports)} successful")
    print("="*40)
    print(f"Output directory: {output_dir}")
    
    return 0 if successful == len(reports) else 1


if __name__ == "__main__":
    sys.exit(main())
