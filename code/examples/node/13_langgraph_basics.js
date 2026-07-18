/**
 * 13_langgraph_basics.js
 * =======================
 * LangGraph 基础 —— 用"状态机"思想构建复杂 AI 工作流
 *
 * LangGraph 是什么？
 *   LangChain 只能做"直线型"流程（A → B → C）。
 *   LangGraph 可以做"图形"流程 —— 有分支、有循环、有条件判断。
 *
 *   类比：
 *   LangChain = 流水线（工序固定，一路向前）
 *   LangGraph = 流程图（可以判断分支、回退重试、条件跳转）
 *
 * 核心概念：
 *   State（状态）：贯穿整个工作流的数据容器，每个节点读取并更新它
 *   Node（节点）：一个处理步骤（AI 调用、工具调用、逻辑判断）
 *   Edge（边）：节点之间的连线（固定方向）
 *   ConditionalEdge（条件边）：根据 State 动态决定走哪条路
 *
 * 涵盖内容：
 *   1. import { StateGraph, Annotation } from '@langchain/langgraph'
 *   2. Annotation.Root 定义 State（类型安全的状态）
 *   3. addNode、addEdge 基础图构建
 *   4. addConditionalEdges 条件分支
 *   5. compile().invoke() 执行图
 *   6. stream() 逐步输出每个节点的状态变化
 *   7. Mermaid 流程图生成说明
 *
 * 运行前：
 *   npm install @langchain/langgraph @langchain/anthropic @langchain/core
 *   export ANTHROPIC_API_KEY=sk-ant-...
 *   node 13_langgraph_basics.js
 */

import { StateGraph, Annotation, END, START } from '@langchain/langgraph';
import { ChatAnthropic } from '@langchain/anthropic';
import { HumanMessage, AIMessage } from '@langchain/core/messages';

// ─────────────────────────────────────────────
// 初始化模型
// ─────────────────────────────────────────────

const model = new ChatAnthropic({
  model: 'claude-opus-4-5',
  maxTokens: 1024,
  apiKey: process.env.ANTHROPIC_API_KEY,
});

// ─────────────────────────────────────────────
// 1. Annotation.Root 定义 State —— 工作流的"共享黑板"
// ─────────────────────────────────────────────

/**
 * State 是什么？
 *   就像一块共享黑板，每个节点都能读取上面的内容，
 *   也能在上面写新内容。工作流执行完毕时，黑板就是最终结果。
 *
 *   类比：接力赛跑的接力棒 —— 每个选手（节点）接过棒（State）、
 *   处理一下，再传给下一个。
 *
 * Annotation.Root 语法要点：
 *   每个字段需要指定 reducer（合并策略）：
 *   - 新值替换旧值：(old, newVal) => newVal ?? old
 *   - 追加到数组：  (old, newItems) => [...old, ...newItems]
 *
 * 为什么需要 reducer？
 *   多个节点可能同时更新同一字段，reducer 决定如何合并这些更新。
 *   类比：多人同时在黑板上写字，reducer 决定谁的字保留下来。
 */
const GraphState = Annotation.Root({
  // 用户输入（新值替换旧值）
  input: Annotation({
    reducer: (old, newVal) => newVal ?? old,
    default: () => '',
  }),

  // AI 的分类结果（新值替换旧值）
  category: Annotation({
    reducer: (old, newVal) => newVal ?? old,
    default: () => '',
  }),

  // 最终回复（新值替换旧值）
  output: Annotation({
    reducer: (old, newVal) => newVal ?? old,
    default: () => '',
  }),

  // 处理步骤记录（追加，不替换）
  // 每个节点完成后追加一条记录，方便调试
  steps: Annotation({
    reducer: (old, newItems) => {
      const items = Array.isArray(newItems) ? newItems : [newItems];
      return [...old, ...items];
    },
    default: () => [],
  }),
});

// ─────────────────────────────────────────────
// 2. 定义节点函数
// ─────────────────────────────────────────────

/**
 * 节点函数的规范：
 *   输入：完整的当前 State
 *   输出：只返回需要更新的字段（部分更新即可）
 *
 * 类比：流水线工人 —— 接到半成品（State），加工局部，传给下一个
 */

// 节点1：分类节点（判断用户输入的问题类型）
async function categorizeNode(state) {
  console.log('  [节点: categorize] 分类中...');

  const response = await model.invoke([
    new HumanMessage(
      `请将下面的问题分类到一个类别（只回答类别名，不要解释）：
      可选类别：技术问题、生活建议、创意写作、数学计算、其他
      问题：${state.input}`
    ),
  ]);

  const category = response.content.trim();
  console.log(`  [节点: categorize] → ${category}`);

  return {
    category,
    steps: [`分类完成：${category}`],
  };
}

// 节点2a：技术问题处理节点
async function handleTechnicalNode(state) {
  console.log('  [节点: handleTechnical] 处理技术问题...');

  const response = await model.invoke([
    new HumanMessage(
      `作为技术专家，请简洁回答（80字以内）：${state.input}`
    ),
  ]);

  return {
    output: `【技术解答】${response.content}`,
    steps: ['技术节点处理完成'],
  };
}

// 节点2b：通用问题处理节点
async function handleGeneralNode(state) {
  console.log('  [节点: handleGeneral] 处理通用问题...');

  const response = await model.invoke([
    new HumanMessage(`请友好地回答（80字以内）：${state.input}`),
  ]);

  return {
    output: `【通用回答】${response.content}`,
    steps: ['通用节点处理完成'],
  };
}

// 节点3：后处理节点（添加免责声明）
async function postProcessNode(state) {
  console.log('  [节点: postProcess] 后处理...');
  return {
    output: state.output + '\n（以上内容由 AI 生成，仅供参考）',
    steps: ['后处理完成'],
  };
}

// ─────────────────────────────────────────────
// 3. 条件路由函数
// ─────────────────────────────────────────────

/**
 * 条件边的路由函数
 *
 * 接收当前 State，返回下一个节点的名称（或 END 结束）。
 * 类比：路口交通警察 —— 根据车牌号（State.category）指挥走哪条路。
 */
function routeByCategory(state) {
  const isTechnical = ['技术问题', '数学计算'].some(cat =>
    state.category.includes(cat)
  );

  const next = isTechnical ? 'handleTechnical' : 'handleGeneral';
  console.log(`  [条件边] category="${state.category}" → ${next}`);
  return next;
}

// ─────────────────────────────────────────────
// 4. 构建并编译图
// ─────────────────────────────────────────────

/**
 * 图的结构（Mermaid 风格描述）：
 *
 *   START → categorize → [条件边]
 *     → handleTechnical → postProcess → END
 *     → handleGeneral   → postProcess → END
 *
 * Mermaid 代码（可粘贴到 https://mermaid.live 查看流程图）：
 *   graph TD
 *     __start__ --> categorize
 *     categorize -->|技术/数学| handleTechnical
 *     categorize -->|其他| handleGeneral
 *     handleTechnical --> postProcess
 *     handleGeneral   --> postProcess
 *     postProcess --> __end__
 */
function buildWorkflow() {
  const workflow = new StateGraph(GraphState);

  // 添加节点：workflow.addNode("节点名", 节点函数)
  workflow.addNode('categorize',       categorizeNode);
  workflow.addNode('handleTechnical',  handleTechnicalNode);
  workflow.addNode('handleGeneral',    handleGeneralNode);
  workflow.addNode('postProcess',      postProcessNode);

  // 添加固定边：START → categorize（入口）
  workflow.addEdge(START, 'categorize');

  // 添加条件边：categorize → (handleTechnical 或 handleGeneral)
  // 第三个参数是路由映射（可省略，用于文档和可视化）
  workflow.addConditionalEdges(
    'categorize',
    routeByCategory,
    {
      handleTechnical: 'handleTechnical',
      handleGeneral:   'handleGeneral',
    }
  );

  // 两个处理节点都流向 postProcess
  workflow.addEdge('handleTechnical', 'postProcess');
  workflow.addEdge('handleGeneral',   'postProcess');

  // postProcess → END（出口）
  workflow.addEdge('postProcess', END);

  // compile() 把图编译成可执行的 Runnable
  return workflow.compile();
}

// ─────────────────────────────────────────────
// 5. 演示：compile().invoke()
// ─────────────────────────────────────────────

async function demo1_invoke() {
  console.log('\n=== 1. compile().invoke() 一次性获取结果 ===');

  const graph = buildWorkflow();

  // 测试1：技术问题
  console.log('\n--- 技术问题 ---');
  const result1 = await graph.invoke({ input: '什么是 Promise？' });
  console.log('输出:', result1.output);
  console.log('步骤:', result1.steps);

  // 测试2：生活问题
  console.log('\n--- 生活问题 ---');
  const result2 = await graph.invoke({ input: '周末去哪里放松比较好？' });
  console.log('输出:', result2.output);
  console.log('步骤:', result2.steps);
}

// ─────────────────────────────────────────────
// 6. 演示：stream() 逐步输出
// ─────────────────────────────────────────────

/**
 * stream() 流式执行
 *
 * 与 invoke() 的区别：
 *   invoke() → 等全部完成，一次性返回最终状态
 *   stream()  → 每个节点完成后，立刻推送该节点产生的状态变化
 *
 * 类比：
 *   invoke() 像等快递到门
 *   stream()  像追踪快递路径，每经过一个城市就收到通知
 *
 * streamMode 选项：
 *   "updates" → 只推送每个节点的状态变化（增量）
 *   "values"  → 每步推送完整的当前 State
 */
async function demo2_stream() {
  console.log('\n=== 2. stream() 逐步输出每个节点状态 ===');

  const graph = buildWorkflow();

  console.log('输入: "如何用 JavaScript 实现快速排序？"\n');

  for await (const stepOutput of graph.stream(
    { input: '如何用 JavaScript 实现快速排序？' },
    { streamMode: 'updates' } // 只推送变化的字段
  )) {
    // stepOutput 格式：{ 节点名: { 状态变化 } }
    for (const [nodeName, stateUpdate] of Object.entries(stepOutput)) {
      console.log(`[${nodeName}] 更新:`, JSON.stringify(stateUpdate));
    }
  }
}

// ─────────────────────────────────────────────
// 7. Mermaid 图生成说明
// ─────────────────────────────────────────────

/**
 * 如何生成 Mermaid 可视化流程图？
 *
 * 步骤：
 *   1. const graph = buildWorkflow();
 *   2. const graphJson = graph.getGraph().toJSON();
 *   3. 把 graphJson 用 LangGraph 的 drawMermaidPng() 转换，
 *      或手动访问 https://mermaid.live 粘贴 Mermaid 代码
 *
 * 本例手动编写的 Mermaid 图：
 */
function demo3_mermaid() {
  console.log('\n=== 3. Mermaid 流程图 ===');
  console.log(`
graph TD
    start([▶ 开始])         --> categorize["分类节点\\n(categorizeNode)"]
    categorize              -->|技术问题/数学计算| tech["技术处理节点\\n(handleTechnical)"]
    categorize              -->|其他类型| general["通用处理节点\\n(handleGeneral)"]
    tech                    --> post["后处理节点\\n(postProcess)"]
    general                 --> post
    post                    --> end_([■ 结束])

  ↑ 将以上内容粘贴到 https://mermaid.live 可看到流程图
`);
}

// ─────────────────────────────────────────────
// 主函数
// ─────────────────────────────────────────────

async function main() {
  console.log('╔══════════════════════════════════════════╗');
  console.log('║  13_langgraph_basics.js  LangGraph 基础  ║');
  console.log('╚══════════════════════════════════════════╝');

  try {
    demo3_mermaid();
    await demo1_invoke();
    await demo2_stream();
  } catch (error) {
    console.error('演示出错:', error.message);
    if (error.message.includes('Cannot find package')) {
      console.error('\n请先安装依赖:');
      console.error('  npm install @langchain/langgraph @langchain/anthropic @langchain/core');
    }
  }

  console.log('\n所有演示完成。');
}

main().catch(err => {
  console.error('未捕获异常:', err.message);
  process.exit(1);
});
