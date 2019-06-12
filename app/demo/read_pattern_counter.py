import numpy as np
import glob

from utils import PatternCounter

def main():
	p = PatternCounter()
	files = glob.glob('tmp/*pkl')
	for file in files:
		p.load(file)
		print(file)
		p.print_list()

if __name__ == '__main__':
	main()