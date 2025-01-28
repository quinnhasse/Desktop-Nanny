import sys
import logging

## error_message_detail gets the error, and produces a message to help fix it.
def error_message_detail(error,error_detail:sys):

    ## provides location and details of error
    _,_,exc_tb=error_detail.exc_info()

    ## gives file name: "8. Errors and Exceptions" Python documentation
    file_name=exc_tb.tb_frame.f_code.co_filename

    #provides message
    error_message="error occured in pyscript name [{0}] line number [{1}] error message[{2}]".format(

        ## variables inside message
        file_name,exc_tb.tb_lineno,str(error))

    return error_message

    


## CustomExceptioninherets exception, takes error message and initializes it to a class variable. Error_detail is being tracked by sys.
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
    
    ## prints the error message
    def __str__(self):
        return self.error_message