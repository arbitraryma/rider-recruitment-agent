import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from rag import retrieve_faq
from site_tools import find_nearest_sites


# ==========================================
# 1. Load environment variables
# ==========================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_URL")
)

MODEL = os.getenv(
    "DASHSCOPE_MODEL",
    "qwen-plus"
)


# ==========================================
# 2. System prompt
# ==========================================

AGENT_SYSTEM_PROMPT = """
你是一个外卖骑手招聘助手。

你可以使用工具查询招聘政策和附近站点。

请严格遵守以下原则：

1. 涉及招聘政策的问题，例如：
   - 是否接受兼职
   - 薪资
   - 年龄要求
   - 工作时间
   - 住宿
   - 电动车
   - 社保
   - 入职条件
   - 发薪日
   - 面试流程

   必须调用 search_faq。

2. 涉及附近站点、最近站点、工作地点的问题，
   如果用户提供经纬度，
   必须调用 find_nearest_sites。

3. 如果问题同时包含招聘政策和站点问题，
   可以同时调用多个工具。

4. 如果检索结果存在不同来源：
   - policy = 正式招聘政策
   - faq = 历史FAQ问答

   若发生冲突，以 policy 为准。

5. 最终回答必须基于工具结果。
   不得编造工具中不存在的信息。

6. find_nearest_sites 只提供地理和距离信息，
   不代表该站点一定正在招聘，
   也不能推断薪资、工作量或站点政策。

7. 回答自然、简洁、明确。
"""


# ==========================================
# 3. Wrapper for RAG tool
# ==========================================

def search_faq(
    query,
    top_k=5
):
    """
    Wrapper around rag.py retrieval pipeline.
    """

    results = retrieve_faq(
        query,
        top_k=top_k
    )

    output = []

    for i, r in enumerate(
        results,
        start=1
    ):

        output.append({
            "rank":
                i,

            "source":
                r.get(
                    "source",
                    "faq"
                ),

            "text":
                r["text"],

            "semantic_score":
                round(
                    r.get(
                        "semantic_score",
                        0
                    ),
                    3
                ),

            "bm25_score":
                round(
                    r.get(
                        "bm25_score",
                        0
                    ),
                    3
                ),

            "rerank_score":
                round(
                    r.get(
                        "rerank_score",
                        0
                    ),
                    3
                )
        })

    return output


# ==========================================
# 4. Tool schemas shown to Qwen
# ==========================================

tools = [

    {
        "type": "function",

        "function": {

            "name":
                "search_faq",

            "description":
                (
                    "Search the official rider recruitment "
                    "policy and FAQ knowledge base. "
                    "Use this for questions about part-time work, "
                    "salary, age requirements, working hours, "
                    "accommodation, vehicles, insurance, benefits, "
                    "onboarding, interviews, or other recruitment policies."
                ),

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {
                        "type": "string",
                        "description":
                            "The user's recruitment-related question."
                    },

                    "top_k": {
                        "type": "integer",
                        "description":
                            "Number of retrieved knowledge chunks.",
                        "default": 5
                    }
                },

                "required": [
                    "query"
                ]
            }
        }
    },


    {
        "type": "function",

        "function": {

            "name":
                "find_nearest_sites",

            "description":
                (
                    "Find the nearest rider recruitment sites "
                    "based on a user's latitude and longitude. "
                    "Returns site location and geographic distance only."
                ),

            "parameters": {

                "type": "object",

                "properties": {

                    "latitude": {
                        "type": "number",
                        "description":
                            "User latitude."
                    },

                    "longitude": {
                        "type": "number",
                        "description":
                            "User longitude."
                    },

                    "top_k": {
                        "type": "integer",
                        "description":
                            "Number of nearest sites to return.",
                        "default": 5
                    }
                },

                "required": [
                    "latitude",
                    "longitude"
                ]
            }
        }
    }
]


# ==========================================
# 5. Execute Qwen tool request
# ==========================================

def execute_tool(
    tool_name,
    arguments
):

    print(
        "\n=========================="
    )

    print(
        "Tool Call"
    )

    print(
        "=========================="
    )

    print(
        "Tool:",
        tool_name
    )

    print(
        "Arguments:",
        arguments
    )


    if tool_name == "search_faq":

        result = search_faq(
            query=arguments[
                "query"
            ],

            top_k=arguments.get(
                "top_k",
                5
            )
        )


    elif tool_name == "find_nearest_sites":

        result = find_nearest_sites(
            latitude=arguments[
                "latitude"
            ],

            longitude=arguments[
                "longitude"
            ],

            top_k=arguments.get(
                "top_k",
                5
            )
        )


    else:

        result = {
            "error":
                f"Unknown tool: {tool_name}"
        }


    print(
        "\nTool Result:"
    )

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2
        )
    )

    return result


# ==========================================
# 6. Agent loop
# ==========================================

def agent_answer(
    query
):

    messages = [

        {
            "role":
                "system",

            "content":
                AGENT_SYSTEM_PROMPT
        },

        {
            "role":
                "user",

            "content":
                query
        }
    ]


    max_steps = 5


    for step in range(
        max_steps
    ):

        print(
            f"\n===== Agent Step {step + 1} ====="
        )


        response = (
            client
            .chat
            .completions
            .create(
                model=MODEL,
                messages=messages,
                tools=tools
            )
        )


        assistant_message = (
            response
            .choices[0]
            .message
        )


        # ----------------------------------
        # No tool call:
        # Qwen has final answer
        # ----------------------------------

        if not (
            assistant_message
            .tool_calls
        ):

            return (
                assistant_message
                .content
            )


        # ----------------------------------
        # Record Qwen's tool request
        # ----------------------------------

        messages.append(
            assistant_message
        )


        # ----------------------------------
        # Execute requested tools
        # ----------------------------------

        for tool_call in (
            assistant_message
            .tool_calls
        ):

            tool_name = (
                tool_call
                .function
                .name
            )


            try:

                arguments = json.loads(
                    tool_call
                    .function
                    .arguments
                )

            except json.JSONDecodeError:

                arguments = {}


            tool_result = execute_tool(
                tool_name,
                arguments
            )


            # ----------------------------------
            # Return tool output to Qwen
            # ----------------------------------

            messages.append({

                "role":
                    "tool",

                "tool_call_id":
                    tool_call.id,

                "content":
                    json.dumps(
                        tool_result,
                        ensure_ascii=False
                    )
            })


    return (
        "抱歉，本轮工具调用次数过多，暂时无法完成。"
    )


# ==========================================
# 7. Terminal chat interface
# ==========================================

if __name__ == "__main__":

    print(
        "\n=============================="
    )

    print(
        "骑手招聘 Agent 启动成功"
    )

    print(
        "=============================="
    )

    print(
        "支持："
    )

    print(
        "- Hybrid RAG"
    )

    print(
        "- BM25 + Semantic Retrieval"
    )

    print(
        "- BGE Reranker"
    )

    print(
        "- Policy / FAQ source handling"
    )

    print(
        "- Nearest-site search"
    )

    print(
        "- Qwen Tool Calling"
    )

    print(
        "\n输入 exit 或 quit 退出"
    )


    while True:

        query = input(
            "\n用户："
        )


        if query.lower() in [
            "exit",
            "quit"
        ]:
            break


        try:

            answer = agent_answer(
                query
            )


            print(
                "\n招聘助手："
            )

            print(
                answer
            )


        except Exception as e:

            print(
                "\n发生错误："
            )

            print(e)