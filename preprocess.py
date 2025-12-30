import spectrum_utils.spectrum as sus
import numpy as np
import argparse
import sys



def selectPeaks(inputfile, outputfile):
    min_mz=50
    max_mz=2500
    min_intensity=0.01
    n_peaks=200
    outf=open(outputfile,"a")
    mz_list=[]
    int_list=[]
    result_str=""
    with open(inputfile,"r") as f:
        for line in f:
            if line.find("BEGIN IONS")!=-1:#
                result_str+=line
            elif line.find("TITLE=")!=-1:#
                result_str+=line
            elif line.find("CHARGE=")!=-1:#
                result_str+=line
            elif line.find("SCANS=")!=-1:#
                result_str+=line
            elif line.find("RTINSECONDS=")!=-1:#
               result_str+=line
            elif line.find("PEPMASS=")!=-1:#
                result_str+=line
            elif line.find("END IONS")!=-1:
                mz_array=np.array(mz_list)
                int_array=np.array(int_list)
                spectrum = sus.MsmsSpectrum("",0,0,mz_array.astype(np.float64),int_array.astype(np.float32))
                try:
                    spectrum.set_mz_range(min_mz, max_mz)
                    if len(spectrum.mz) == 0:
                        raise ValueError
                    spectrum.filter_intensity(min_intensity, n_peaks)
                    # spectrum.filter_intensity(0.0, n_peaks)

                    if len(spectrum.mz) == 0:
                        raise ValueError
                    spectrum.scale_intensity("root", 1)
                    intensities = spectrum.intensity / np.linalg.norm(
                        spectrum.intensity
                    )
                except ValueError:
                    # Replace invalid spectra by a dummy spectrum.
                    print(ValueError)
                    sys.exit(1)
                mz_list=[]
                int_list=[]
                if len(spectrum.mz)!=len(intensities):
                    print("length not equ")
                    sys.exit(1)
                for id in range(len(intensities)):
                    result_str+=str(spectrum.mz[id])+" "+str(intensities[id])+"\n"
                result_str+=line
                outf.write(result_str)
                result_str=""
            elif line=="\n":
                continue
            else:
                tmp=line.strip().split(" ")
                try:
                    mz_list.append(float(tmp[0]))
                    int_list.append(float(tmp[1]))
                except: 
                    print(tmp)
    f.close()

    outf.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile",type=str,default="",help='the input mgf file path')
    parser.add_argument("--outfile",type=str,default="",help='the output mgf file path')
    args = parser.parse_args()
    selectPeaks(args.infile, args.outfile)