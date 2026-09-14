import aiohttp
import os
from urllib.parse import urlencode
import json
import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.DEBUG)


async def plex_login():
    url = "https://plex.tv/api/v2/pins"
    headers = {
        "X-Plex-Client-Identifier": os.getenv("PLEX_CLIENT_ID"),
        "X-Plex-Product": os.getenv("PLEX_CLIENT_NAME"),
        "accept": "application/json",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, headers=headers, params={"strong": "true"}
        ) as response:
            if response.status == 201:
                data = await response.json()
            else:
                raise Exception(f"Failed to create PIN: {response.status}")

    logging.debug(f"PIN creation response data: {json.dumps(data, indent=2)}")
    code = data["code"]
    pin_id = data["id"]
    logging.info(f"Created PIN with code: {code} and id: {pin_id}")
    base_login_url = "https://app.plex.tv/auth"
    query_params = {
        "clientID": os.getenv("PLEX_CLIENT_ID"),
        "code": code,
        "context[device][product]": os.getenv("PLEX_CLIENT_NAME"),
    }
    login_url = f"{base_login_url}#?{urlencode(query_params)}"
    print(f"Please visit the following URL to authenticate:\n{login_url}")
    while True:
        await asyncio.sleep(1)
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/{pin_id}", headers=headers) as response:
                if response.status == 200:
                    pin_data = await response.json()
                    if pin_data.get("authToken"):
                        auth_token = pin_data["authToken"]
                        print(f"Authentication successful! Auth Token: {auth_token}")
                        return auth_token
                else:
                    pass

    print(f"Your new token is: {auth_token}")


if __name__ == "__main__":
    asyncio.run(plex_login())
