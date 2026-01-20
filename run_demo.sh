#!/bin/bash
# Starting
# python agent/run_agent.py --input_file agent/demos/demo_inputs.jsonl
# python agent/run_agent.py --input "Count words: hello world This is James"

# TSR
python agent/run_agent.py --input "Run TSR on image ./sample_table.png. Use backend='tsr'. Return HTML only."
python agent/run_agent.py --input "Extract OTSL from ./sample_table.png using backend='tsr'. Return OTSL only."
python agent/run_agent.py --input "Extract table from ./sample_table.png using backend='tsr'. Convert to HTML and save to outputs/sample_table.html."
python agent/run_agent.py --input "Call tsr_extract with image_path='./sample_table.png', backend='tsr', output_format='html'. Then call save_html to write outputs/sample_table.html."
python agent/run_agent.py --input "Extract table from ./sample_table.png using backend='tsr'. Convert to HTML and save to outputs/sample_table.html."
python agent/run_agent.py --input "Extract table from ./sample_table.png using backend='perception'. Convert to HTML and save to outputs/sample_table.html."
python agent/run_agent.py --input --debug "Extract table from ./sample_table.png using backend='tsr'. Convert to HTML and save to outputs/sample_table.html."
python agent/run_agent.py --input --trace "Extract table from ./sample_table.png using backend='perception'. Convert to HTML and save to outputs/sample_table.html."