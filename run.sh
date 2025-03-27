# python 0_filter_small_pdfs.py --input_dir ./data_input/osint_all --output_dir ./data_input/osint_small 

sh 1_pdf_to_md.sh ./data_input/osint_small ./data_input/osint_md

python 2_md_to_kg.py --input_dir ./data_input/osint_md --output_file ./data_output/osint_small

python 3_postprocess_kg.py --kg ./data_output/osint_small --output data_output/osint_small
