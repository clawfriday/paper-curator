"""Slack helper utilities."""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException


async def resolve_slack_channel_id(client, channel_input: str) -> str:
    """Resolve a Slack channel name/URL/ID to channel ID."""
    if channel_input.startswith("C") and len(channel_input) > 8:
        return channel_input

    if channel_input.startswith("https://"):
        parts = channel_input.split("/")
        for part in parts:
            if part.startswith("C") and len(part) > 8:
                return part

    channel_name = channel_input.lstrip("#")

    try:
        response = await asyncio.to_thread(client.conversations_list, limit=1000)
        if not response["ok"]:
            raise HTTPException(
                status_code=400,
                detail=f"Slack API error: {response.get('error', 'Unknown error')}",
            )
        channels = response.get("channels", [])
        for channel in channels:
            if channel["name"] == channel_name:
                return channel["id"]

        available_channels = [ch["name"] for ch in channels[:20]]
        error_msg = f"Channel '{channel_input}' not found in accessible channels. "
        error_msg += f"Available channels (first 20): {', '.join(available_channels)}. "
        error_msg += "\n\nTroubleshooting:\n"
        error_msg += "1. Verify the channel name is correct (check for typos, spaces, or different formatting)\n"
        error_msg += "2. IMPORTANT: If using a User OAuth Token (xoxp-...), the USER whose token it is must be a member of the channel.\n"
        error_msg += "   Adding the app/bot to the channel is not enough - the user must also join the channel.\n"
        error_msg += "3. Alternatively, use a Bot Token (xoxb-...) and ensure the bot is added to the channel\n"
        error_msg += "4. Check if the channel is archived (archived channels may not appear)\n"
        error_msg += "5. Verify you're using the token for the correct Slack workspace\n"
        error_msg += "6. Try using the channel ID instead: Right-click channel → Copy link → Use the C1234567890 ID from the URL"
        raise HTTPException(status_code=404, detail=error_msg)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resolving Slack channel: {e}")


async def fetch_slack_messages(client, channel_id: str) -> list[dict[str, Any]]:
    """Fetch all messages from a Slack channel with pagination."""
    all_messages = []
    cursor = None

    while True:
        try:
            response = await asyncio.to_thread(
                client.conversations_history,
                channel=channel_id,
                cursor=cursor,
                limit=200,
            )
            if not response["ok"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Slack API error: {response.get('error', 'Unknown error')}",
                )
            messages = response.get("messages", [])
            all_messages.extend(messages)
            cursor = response.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                break
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching Slack messages: {e}")

    return all_messages
