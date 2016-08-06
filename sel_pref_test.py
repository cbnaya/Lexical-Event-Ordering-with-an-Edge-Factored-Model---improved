'''
Created on 2 Oct 2013

@author: oabend
'''
import sel_pref_model, sys, pickle, os
from optparse import OptionParser

def cmd_line_parser():
    "Returns the command line parser"
    usage = "usage: %prog [options]\n"
    opt_parser = OptionParser(usage=usage)
    opt_parser.add_option("-m", action="store", type="string", dest="model",
                          help="the model (COMP/VOBJ)")
    opt_parser.add_option("-i", action="store", type="string", dest="input",
                          help="input file")
    opt_parser.add_option("-o", action="store", type="string", dest="output",
                          help="output file")
    return opt_parser


if __name__ == '__main__':
    parser = cmd_line_parser()
    (options, args) = parser.parse_args(sys.argv)
    if options.model == None or options.input == None or options.output == None:
        parser.error("options -o, -i, -m must be specified")
    
    if options.model == "COMP":
        sp_model = sel_pref_model.SelPrefModel('COMP')
    elif options.model == "VOBJ":
        sp_model = sel_pref_model.SelPrefModel('VOBJ')
    else:
        parser.error("model not yet implemented")
    
    sp_model.update_from_dep_data(options.input)

    print(sp_model)
    
    f_out = open(options.output, 'wb')
    pickle.dump(sp_model, f_out)
    f_out.close()

