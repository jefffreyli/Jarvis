# Jarvis relay

Lets Alexa and Linq reach the Jarvis backend without a tunnel. The backend opens an
outbound websocket to API Gateway; the Lambdas here forward requests down it.

```
Echo -> Alexa -> SkillFunction --\
                                  +-> WebSocket API -> backend
Linq -> HTTP API -> LinqWebhookFunction --/          <- replies (ConnectionsFunction -> DynamoDB)
```

## Setup

1. Install the tools and sign in: `brew install awscli aws-sam-cli`, then `aws configure`.
2. Make a shared secret: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`.
3. Create the skill at https://developer.amazon.com/alexa/console/ask (Custom model,

   "Provision your own" backend). In the JSON Editor paste `alexa-interaction-model.json`,
     then save and build. Copy the skill id (`amzn1.ask.skill...`).
4. Deploy from this directory: `sam build && sam deploy --guided`, entering the secret

   and the skill id. Later deploys are just `sam build && sam deploy`.
5. In the skill's Endpoint page, choose AWS Lambda ARN and paste the `SkillFunctionArn` output.
6. Add to the project's `.env`:
   ```
     RELAY_URL=<RelayUrl output>
     RELAY_SECRET=<the secret>
     LINQ_WEBHOOK_URL=<LinqWebhookUrl output>   # only needed for texting
   ```
7. Run `python3 main.py` in `backend/` and say "Alexa, open Jarvis".

Alexa gives a skill 8 seconds to answer. When Jarvis takes longer, say "any update" to hear the answer.
