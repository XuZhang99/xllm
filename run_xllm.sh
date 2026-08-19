curl -s "http://127.0.0.1:29000/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer <API Key>" \
    -d '{
          "model": "GLM-5.2-w8a8",
          "messages": [
            {"role": "system", "content": "You are a user assistant."},
            {"role": "user", "content": "介绍下北京"}
          ],
          "top_p": 0.95,
          "temperature": 0.6,
          "top_k": -1,
          "stream": false
        }'