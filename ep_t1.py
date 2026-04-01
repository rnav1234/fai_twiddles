import requests,argparse
import os,sys

token = os.environ.get("FRIENDLI_TOKEN") or "YOUR_FRIENDLI_TOKEN"

url = "https://api.friendli.ai/dedicated/v1/chat/completions"

headers = {
  "Authorization": "Bearer " + token,
  "Content-Type": "application/json"
}

payload = {
  "model": "depspuskbpwcov3",
  "messages": [
    {
      "role": "user",
      "content": ""
    }
  ],
  "max_tokens": 16384,
  "temperature": 0,
  "top_p": 0.95,
#  "stream": True,
#  "stream_options": {
#    "include_usage": True
#  }
}


def main():
   parser = argparse.ArgumentParser(
        description='FriendliAI EP tester',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument('-q','--query', required=True, help='query string')
   parser.add_argument('-m','--model', required=True, help='FAI endpoint_id or payload["model"]')
   parser.add_argument('-r','--reasoning', help='reasoning effort: low,medium or high')
   args = parser.parse_args()
   payload['model'] = args.model
   payload['messages'][0]['content'] = args.query
   if args.reasoning:
      if args.reasoning in ('low','medium','high'):
         payload['reasoning_effort'] = args.reasoning
      else:
         print(f'Invalid -r reasoning value [{args.reasoning}]. Expected one of low,medium,high',file=sys.stderr)
         sys.exit(1)
   #import sys; print(f"XXXXX [{payload = }]",file=sys.stderr)
   response = requests.request("POST", url, json=payload, headers=headers)
   #print(response.text)

   # Check response if 'reasoning' was returned
   if response.status_code == 200:
        result = response.json()
        message = result['choices'][0]['message']

        # Qwen3 models return reasoning in a specific field if supported by the API
        reasoning = message.get('reasoning_content', "No reasoning trace returned.")
        answer = message.get('content')

        print(f"THINKING PHASE:\n{reasoning}\n")
        print(f"FINAL ANSWER:\n{answer}")
   else:
        print(f"Error {response.status_code}: {response.text}")

if __name__ == '__main__':
   main()
