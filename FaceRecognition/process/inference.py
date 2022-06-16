import sys, os, time
sys.path.append(os.path.join(os.getcwd(), "process"))

import utils
from process.respondent import respondent


def get_result_attendance(attendance_info):
    
    result = {
        "attendance_info": attendance_info,
    }

    return result

def inference_face(url):
    respondent.response(url)
    # if result is None:
    #     return None
    
    # attendance_info = result

    # final_result = get_result_attendance(attendance_info)
    # return final_result
    
    
def inference_api(url):

    # if not utils.check_validation.is_valid_url(url):
    #     return "đường link không hợp lệ, vui lòng thử lại"
        
    # start = time.time()
    inference_face(url)
    # print(time.time() - start)

    # if result is None:
    #     return "Không hợp lệ, vui lòng thử lại"

    # return result
