RelayClient
- The RelayClient is a websocket client on your Mac. It dials out to an API Gateway WebSocket API (the server end), and AWS sends Alexa and Linq requests back down that same connection.
- The problem a relay solves is that Alexa skills live on the internet. When you make a request to Alexa through a voice command, they need to send an HTTP request to a public URL. Jarvis runs on your Mac locally so there's no connection to your agent.
- The relay here is a websocket server between an API gateway websocket and your local agent on your Mac.
- RelayClient is hosted as a backend process in this local agent. It is the client end of the websocket connection.

AWS
- WebSocket API: Local backend connects to
- Lambdas
    - ConnectionsFunctions handles every Websocket request
    - SkillFunction handles the Alexa skill's endpoint. 
- API Gateway: managed AWS service on the us-east-1 region. Configured through the CloudFormation template.
    - API Gateway is the public front door to your AWS backend. It takes internet traffic and turns it into Lambda calls. Your Lambdas can't be reached from the internet on their own and only run when something invokes them, so API Gateway sits between them and the outside world.
    - holds the websocket open
    - routes requests to lambda functions
    - lambdas push their response over the websocket to local backend service

Intent
- An intent is Alexa's label for what the user wants to do. Alexa doesn't send your Lambda the raw audio, or even reliably the full transcript. It converts speech to text, matches that text against intents you declared ahead of time, and sends your Lambda the name of the matching intent.
- JarvisIntent: passes the entire query to the agent
- UpdateIntent: lets the agent work in the background and ask Alexa for the update when the task takes longer than 8 seconds, which is Alexa's limit. Alexa can't automatically reply so this is a work around. The message is saved into a queue in the db so when someone asks for an update it will retrieve from there.