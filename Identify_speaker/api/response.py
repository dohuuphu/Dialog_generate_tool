import json
from starlette.responses import Response

class APIResponse():
        def json_format(success, content, status_code, response=None, time_process=None, score = None, speaker=None):
              
                return  Response(content=json.dumps(
                                {
                                'meta':{
                                        'success': success,
                                        'msg':content
                                        },
                                'response': response if response else None,
                                'time_process': time_process if time_process else None,
                                'score': score if score else None,
                                'speaker': speaker if speaker else None
                                },
                                ),
                        status_code=status_code,
                        media_type="application/json"
                        )
