import requests,argparse
import os

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
  "stream": True,
  "stream_options": {
    "include_usage": True
  }
}


def main():
   parser = argparse.ArgumentParser(
        description='FriendliAI EP tester',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument('-q','--query', required=True, help='query string')
   parser.add_argument('-m','--model', required=True, help='FAI endpoint_id or payload["model"]')
   args = parser.parse_args()
   payload['model'] = args.model
   payload['messages'][0]['content'] = args.query
   import sys; print(f"XXXXX [{payload = }]",file=sys.stderr)
   response = requests.request("POST", url, json=payload, headers=headers)
   print(response.text)

if __name__ == '__main__':
   main()
