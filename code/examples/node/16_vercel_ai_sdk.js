/**
 * 16_vercel_ai_sdk.js
 * Vercel AI SDK 入门 —— 对前端工程师最友好的 AI 调用方式
 *
 * 为什么前端开发者应该优先学这个？（见文末第 9 节）
 *
 * 安装依赖：
 *   npm install ai @ai-sdk/anthropic zod
 *
 * 如果用 OpenAI 模型：
 *   npm install ai @ai-sdk/openai
 *
 * 运行前先设置环境变量：
 *   export ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
 *   node 16_vercel_ai_sdk.js
 */

// ─── 依赖导入 ───────────────────────────────────────────────────────────────

const { generateText, streamText, generateObject, tool } = require("ai");
const { createAnthropic } = require("@ai-sdk/anthropic");
const { z } = require("zod");

// 初始化 Anthropic provider
// ANTHROPIC_API_KEY 会自动从环境变量读取
const anthropic = createAnthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

// 选定模型 —— 换模型只需改这一行
const model = anthropic("claude-3-5-haiku-20241022");

// ─── 1. generateText：最简单的一次性调用 ────────────────────────────────────
//
// 类比：就像 fetch() 发请求，等服务器返回完整响应后再处理
// 适合：分类、翻译、短文生成等不需要流式效果的场景

async function demo_generateText() {
  console.log("\n======== 1. generateText（最简单调用）========");

  const result = await generateText({
    model,
    prompt: "用一句话解释什么是 TypeScript。",
  });

  // result.text 就是模型的回答字符串，直接用即可
  console.log("回答:", result.text);

  // 顺便看看消耗了多少 token（日后估算费用用得到）
  console.log("Token 用量:", result.usage);
  // { promptTokens: 12, completionTokens: 30, totalTokens: 42 }
}

// ─── 2. generateText + system prompt：给 AI 设定角色 ──────────────────────
//
// system 参数相当于"指令说明书"，在对话开始前告诉 AI 它是谁、该怎么回答

async function demo_systemPrompt() {
  console.log("\n======== 2. System Prompt（角色设定）========");

  const result = await generateText({
    model,
    system: "你是一位专门教小学生编程的老师，只用最简单的语言回答，不超过 3 句话。",
    messages: [
      { role: "user", content: "什么是变量？" },
      { role: "assistant", content: "变量就像一个贴了标签的盒子，用来存东西。" },
      { role: "user", content: "那常量呢？" }, // 多轮对话：传入历史消息
    ],
    maxTokens: 200,    // 最多生成多少 token
    temperature: 0.7,  // 0 = 稳定确定，1 = 更有创意，默认 1
  });

  console.log("回答:", result.text);
}

// ─── 3. streamText：流式输出 ─────────────────────────────────────────────────
//
// 类比：类似 ReadableStream，数据像水流一样逐字返回
// 前端用途：打字机效果、实时翻译、长文章生成
// 在 Next.js 里配合 useChat Hook，几行代码就有流式对话界面

async function demo_streamText() {
  console.log("\n======== 3. streamText（流式输出）========");
  process.stdout.write("流式回答: ");

  const result = streamText({
    model,
    prompt: "给我列出 3 条学习 React 的建议，每条一行。",
  });

  // for-await 逐块读取，适合终端和 Node.js 后端场景
  for await (const chunk of result.textStream) {
    process.stdout.write(chunk); // 实时打印，不换行，产生打字机效果
  }

  console.log(); // 最后补换行
  const usage = await result.usage;
  console.log("共消耗 token:", usage.totalTokens);
}

// ─── 4. generateObject：结构化输出（配合 Zod）────────────────────────────────
//
// 前端最爱！不用手动 JSON.parse + 错误处理
// AI 保证输出符合你定义的 Zod Schema，类型安全，直接用
// TypeScript 项目里 result.object 自动有完整的类型推断

async function demo_generateObject() {
  console.log("\n======== 4. generateObject（结构化输出 + Zod）========");

  // 第一步：用 Zod 定义期望的数据结构
  // Zod 是前端项目里常见的运行时类型校验库，和 TypeScript 天然配套
  const ProductSchema = z.object({
    name: z.string().describe("产品名称"),
    price: z.number().describe("价格，单位：元"),
    features: z.array(z.string()).describe("核心功能列表，3 条"),
    targetUser: z.string().describe("目标用户群体"),
    isRecommended: z.boolean().describe("是否推荐给初学者"),
  });

  // 第二步：传入 schema，generateObject 自动保证输出符合结构
  const { object } = await generateObject({
    model,
    schema: ProductSchema,
    prompt: "帮我虚构一款面向前端工程师的 AI 编程助手产品信息。",
  });

  // object 是经过 Zod 校验的 JS 对象，类型安全
  console.log("结构化数据:", JSON.stringify(object, null, 2));
  console.log("price 类型验证:", typeof object.price); // number，不是字符串 "50"
}

// ─── 5. tools 参数：工具调用（Function Calling）──────────────────────────────
//
// 工具调用让 AI 能"做事"，而不只是"说话"
// 类比：给 AI 配了工具箱，它能自己决定用哪个工具完成任务
// 写法和 React 组件的 props 定义非常像，前端上手零门槛

async function demo_tools() {
  console.log("\n======== 5. 工具调用（tools 参数）========");

  const result = await generateText({
    model,
    tools: {
      // tool() 是 AI SDK 的辅助函数，自动把 Zod schema 转成工具格式
      get_weather: tool({
        description: "获取指定城市的当前天气信息",
        parameters: z.object({
          city: z.string().describe("城市名称，例如：北京、上海"),
          unit: z
            .enum(["celsius", "fahrenheit"])
            .describe("温度单位")
            .optional(),
        }),
        // execute 是工具的实际执行函数
        execute: async ({ city, unit = "celsius" }) => {
          console.log(`  [工具调用] 查询 ${city} 天气，单位: ${unit}`);
          // 真实项目里这里调用真实天气 API
          return { city, temperature: 22, unit, condition: "晴天", humidity: "60%" };
        },
      }),

      calculate: tool({
        description: "执行数学计算，返回计算结果",
        parameters: z.object({
          expression: z.string().describe("数学表达式，例如：2 + 3 * 4"),
        }),
        execute: async ({ expression }) => {
          console.log(`  [工具调用] 计算: ${expression}`);
          try {
            // 注意：生产环境请用安全的数学库，这里仅作演示
            const result = Function(`"use strict"; return (${expression})`)();
            return { expression, result };
          } catch {
            return { expression, result: "计算错误" };
          }
        },
      }),
    },
    prompt: "北京今天天气怎么样？另外帮我算一下 (25 + 75) * 2 等于多少？",
    maxSteps: 5, // 允许 AI 多次调用工具（见下一节说明）
  });

  console.log("最终回答:", result.text);
  console.log("执行步骤数:", result.steps.length);
}

// ─── 6. maxSteps：自动多步循环（Agent 的核心机制）────────────────────────────
//
// maxSteps 的运作原理：
//   用户输入 → AI 思考 → 调用工具 → 拿到结果 → AI 继续思考 → ... → 最终回答
//   设置 maxSteps: N，就是允许上面的循环最多执行 N 次
//
// 这是把 AI 变成"Agent"的关键：AI 不是被动回答，而是主动规划和执行

async function demo_maxSteps() {
  console.log("\n======== 6. maxSteps（自动多步 Agent 循环）========");

  let stepCount = 0;

  const result = await generateText({
    model,
    tools: {
      search_user: tool({
        description: "在数据库中根据 ID 查询用户信息",
        parameters: z.object({
          userId: z.string().describe("用户 ID"),
        }),
        execute: async ({ userId }) => {
          stepCount++;
          console.log(`  [步骤 ${stepCount}] 查询用户: ${userId}`);
          // 模拟数据库查询
          return { userId, name: "张三", email: "zhang@example.com", age: 28 };
        },
      }),

      send_email: tool({
        description: "给指定邮箱发送邮件",
        parameters: z.object({
          to: z.string().describe("收件人邮箱"),
          subject: z.string().describe("邮件主题"),
          body: z.string().describe("邮件内容"),
        }),
        execute: async ({ to, subject, body }) => {
          stepCount++;
          console.log(`  [步骤 ${stepCount}] 发送邮件给 ${to}: ${subject}`);
          return { success: true, messageId: "msg_001" };
        },
      }),
    },
    maxSteps: 5,
    // onStepFinish 每步完成时触发，可用于进度显示、日志记录
    onStepFinish: ({ stepType, toolCalls }) => {
      if (toolCalls?.length) {
        console.log(`  -> 本步调用了工具: ${toolCalls.map((t) => t.toolName).join(", ")}`);
      }
    },
    prompt:
      "查询用户 ID 为 U001 的信息，然后给他发一封欢迎邮件，邮件内容要亲切友好。",
  });

  console.log("最终回答:", result.text);
  console.log("总步骤数:", result.steps.length);
}

// ─── 7. React useChat Hook（注释展示，需在 React 项目中使用）──────────────────
//
// useChat 是 Vercel AI SDK 提供的 React Hook
// 几行代码搞定：消息列表管理、流式输入、加载状态、错误处理
// 需配合后端 API Route（见第 8 节）

/*
// ---- components/ChatBox.tsx ----
"use client";
import { useChat } from "ai/react";

export default function ChatBox() {
  const {
    messages,           // 消息列表 [{id, role: "user"|"assistant", content}]
    input,              // 当前输入框的值（受控）
    handleInputChange,  // 绑定到 <input onChange>
    handleSubmit,       // 绑定到 <form onSubmit>
    isLoading,          // 是否正在等待 AI 回复
    error,              // 错误对象，可用于展示错误提示
    stop,               // 中止当前流式输出
    reload,             // 重新发送最后一条用户消息
    setMessages,        // 手动修改消息列表（如清空对话）
  } = useChat({
    api: "/api/chat",      // 后端 Route Handler 地址
    initialMessages: [],   // 初始消息（可用来设置欢迎语）
    onFinish: (message) => {
      // AI 回复完成时触发
      console.log("AI 回复完成:", message.content);
    },
    onError: (error) => {
      console.error("出错了:", error);
    },
  });

  return (
    <div className="flex flex-col h-screen">
      // 消息列表区域
      <div className="flex-1 overflow-y-auto p-4">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`mb-4 ${msg.role === "user" ? "text-right" : "text-left"}`}
          >
            <span className={`inline-block p-3 rounded-lg ${
              msg.role === "user" ? "bg-blue-500 text-white" : "bg-gray-100"
            }`}>
              {msg.content}
            </span>
          </div>
        ))}
        {isLoading && <div className="text-gray-400">AI 思考中...</div>}
        {error && <div className="text-red-500">出错了: {error.message}</div>}
      </div>

      // 输入区域
      <form onSubmit={handleSubmit} className="p-4 border-t flex gap-2">
        <input
          value={input}
          onChange={handleInputChange}
          placeholder="输入你的问题..."
          disabled={isLoading}
          className="flex-1 border rounded-lg p-2"
        />
        <button type="submit" disabled={isLoading || !input.trim()}>
          发送
        </button>
        {isLoading && (
          <button type="button" onClick={stop}>
            停止
          </button>
        )}
      </form>
    </div>
  );
}
*/

// ─── 8. Next.js App Router API Route 示例（注释展示）─────────────────────────
//
// 这是配合 useChat Hook 的后端部分
// 前端 useChat 自动把消息 POST 到这里，这里调用 AI 并返回流式响应

/*
// ---- app/api/chat/route.ts ----
import { streamText } from "ai";
import { createAnthropic } from "@ai-sdk/anthropic";

const anthropic = createAnthropic();
const model = anthropic("claude-3-5-haiku-20241022");

// Next.js App Router 的 POST 处理函数
export async function POST(req: Request) {
  // useChat 自动发送 { messages: [...] } 格式的 JSON
  const { messages } = await req.json();

  const result = streamText({
    model,
    system: "你是一个友好的 AI 助手，用中文回答。",
    messages,  // 传入完整对话历史，支持多轮对话
  });

  // toDataStreamResponse() 把流转成 useChat 能理解的格式
  // 一行代码搞定流式响应格式化
  return result.toDataStreamResponse();
}

// ──────────────────────────────────────────────
// 换模型只需改 2 行（其余代码一行不动）：
// ──────────────────────────────────────────────
// 换成 OpenAI GPT-4o：
//   import { createOpenAI } from "@ai-sdk/openai";
//   const openai = createOpenAI();
//   const model = openai("gpt-4o-mini");
//
// 换成 Google Gemini：
//   import { createGoogleGenerativeAI } from "@ai-sdk/google";
//   const google = createGoogleGenerativeAI();
//   const model = google("gemini-1.5-flash");
*/

// ─── 9. 为什么前端开发者应该优先学 Vercel AI SDK ──────────────────────────────
//
// 一、统一 API，一次学会，所有模型通用
//   - 支持 Anthropic / OpenAI / Google / Mistral / Cohere 等 20+ 家提供商
//   - 换模型只改一行 import + 模型名，其余代码完全不动
//   - 不用为每家 AI 厂商单独学习 SDK
//
// 二、React Hook 开箱即用，零配置接入
//   - useChat：完整聊天界面，消息列表+流式+加载状态全包
//   - useCompletion：文本补全 UI
//   - useObject：流式结构化输出（边生成边展示）
//   - 不用自己写 SSE 解析、消息队列、错误重试
//
// 三、TypeScript 类型推断极致友好
//   - generateObject + Zod：result.object 自动拥有完整类型
//   - 工具参数类型、返回值类型全部推断，IDE 提示完善
//   - 不用写任何 as any 或 type assertion
//
// 四、前后端共用同一套概念
//   - generateText / streamText 在 Node.js 和 Edge Runtime 都能跑
//   - Next.js Route Handler 里就几行代码，配合 useChat 天然契合
//   - 全栈 JS 工程师不需要在两套体系间切换
//
// 五、生态和工具链成熟
//   - Vercel AI SDK 由 Vercel 官方维护，版本稳定，更新积极
//   - 和 Next.js / v0 / Vercel 平台深度集成
//   - 大量开源模板可直接参考：npx create-next-app@latest --example ai-chatbot
//
// 总结一句话：
//   如果你已经会 React + TypeScript，学 Vercel AI SDK
//   等于用熟悉的写法，获得完整的 AI 能力，学习曲线最平。

// ─── 主程序 ──────────────────────────────────────────────────────────────────

async function main() {
  console.log("=== Vercel AI SDK 入门示例 ===");
  console.log("模型: claude-3-5-haiku-20241022\n");

  if (!process.env.ANTHROPIC_API_KEY) {
    console.error("错误: 请先设置 ANTHROPIC_API_KEY 环境变量");
    console.error("  export ANTHROPIC_API_KEY=sk-ant-xxxxxxxx");
    process.exit(1);
  }

  try {
    await demo_generateText();
    await demo_systemPrompt();
    await demo_streamText();
    await demo_generateObject();
    await demo_tools();
    await demo_maxSteps();

    console.log("\n=== 全部示例运行完成 ===");
    console.log("\n下一步推荐:");
    console.log("1. 查看 17_ai_sdk_advanced.js 了解流式对象、中间件等高级特性");
    console.log("2. 用官方模板快速创建完整应用:");
    console.log("   npx create-next-app@latest --example ai-chatbot my-chat-app");
    console.log("3. 官方文档: https://sdk.vercel.ai/docs");
  } catch (error) {
    const msg = error?.message ?? String(error);
    if (msg.includes("API key") || msg.includes("api_key") || msg.includes("401")) {
      console.error("\n错误: ANTHROPIC_API_KEY 无效或未设置");
      console.error("  export ANTHROPIC_API_KEY=sk-ant-xxxxxxxx");
    } else {
      console.error("\n运行出错:", msg);
    }
    process.exit(1);
  }
}

main();
