import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_prefix', type=str, default='test', required=False, help='The prefix of the input file.')
parser.add_argument('-o', '--output_suffix', type=str, default='output', required=False, help='The suffix of the output file.')
parser.add_argument('-s', '--starting_page_number', type=int, default=40, required=False, help='The starting page number.')
parser.add_argument('-e', '--ending_page_number', type=int, default=430, required=False, help='The ending page number.')

args = parser.parse_args()
start_dire = "./main.py"

for i in range(args.starting_page_number,args.ending_page_number+1):
    input_name = args.input_prefix + '_' + str(i) + '.png'
    output_name = args.input_prefix + '_' + str(i) + '_' + args.output_suffix + '.png'
    print("Picture No." + str(i))
    os.system("python %s -i \"%s\" -o \"%s\"" %(start_dire,input_name,output_name)) # 传入参数并运行
    print()