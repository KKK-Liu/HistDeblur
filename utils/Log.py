import logging

class Logger():
    def __init__(self, args) -> None:
        self.args = args
        
        Format = "%(asctime)s line: %(lineno)s %(message)s"
        level = [logging.DEBUG, logging.INFO,
                logging.WARNING, logging.ERROR, logging.CRITICAL]
        
        logging.basicConfig(level=level[1], format=Format,
                            filename=args.logfilename+args.name+'.log', filemode=args.logfilemode)
        logging.info("--------------start---------------")
        
    def __call__(self, msg):
        logging.info(msg)
        