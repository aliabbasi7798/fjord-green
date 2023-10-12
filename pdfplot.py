import os
import PyPDF2
import cairosvg

def svg_to_pdf(svg_path, pdf_path):
    with open(svg_path, 'rb') as svg_file:
        svg_data = svg_file.read()
        cairosvg.svg2pdf(file_obj=svg_data, write_to=pdf_path)

def merge_pdfs(pdf_files, output_path):
    merger = PyPDF2.PdfFileMerger()

    for pdf_file, label in pdf_files:
        merger.append(pdf_file, bookmark=label)

    merger.write(output_path)
    merger.close()

def main():
    # Get input SVG files
    svg_files = input("final_plots/alpha=0.01_clusters_E=1_cost.svg,final_plots/alpha=0.01_clusters_E=1_cost.svg,final_plots/alpha=0.01_clusters_E=1_cost.svg,final_plots/alpha=0.01_clusters_E=1_cost.svg").split(',')

    # Get labels for each SVG file
    labels = input("a,b,c,d").split(',')

    # Combine SVG files with labels into a list of tuples
    svg_files_with_labels = list(zip(svg_files, labels))
    print("hi")
    # Convert SVG to PDF
    pdf_files = []
    for svg_file, label in svg_files_with_labels:
        pdf_file = os.path.splitext(svg_file)[0] + '.pdf'
        svg_to_pdf(svg_file.strip(), pdf_file)
        pdf_files.append((pdf_file, label.strip()))
    print("hi")
    # Merge PDFs into a single file
    output_pdf = input("final_plots/merged_output.pdf").strip()
    merge_pdfs(pdf_files, output_pdf)
    print("hi")
    # Clean up intermediate PDF files
    for pdf_file, _ in pdf_files:
        os.remove(pdf_file)

    print(f"Merged PDF saved to {output_pdf}")

if __name__ == "__main__":
    main()