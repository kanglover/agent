/**
 * 15_langgraph_multiagent.js
 * ===========================
 * LangGraph 多 Agent 系统 —— 团队协作完成复杂任务
 *
 * 什么是多 Agent 系统？
 *   让多个专门的 AI（各有专长）协同工作，就像一个小团队：
 *   - 一个"项目经理"（Supervisor）负责分配任务、决定顺序
 *   - 多个"专家"（Worker）各司其职
 *
 *   类比：软件开发团队
 *   - researcher（调研员）：搜集资料和背景信息
 *   - coder（开发员）：根据需求写代码
 *   - reviewer（审查员）：审查代码质量和正确性
 *   - supervisor（PM）：决定该让谁先干活，汇总最终结果
 *
 * 涵盖内容：
 *   1. 共享 State（messages、next、results 等字段）
 *   2. Supervisor Agent（路由决策节点）
 *   3. 3个 Worker：researcher、coder、reviewer
 *   4. 路由函数（条件边，根据 next 字段跳转）
 *   5. 完整演示：researcher 搜背景 → coder 写代码 → reviewer 检查
 *
 * 运行前：
 *   npm install @langchain/langgraph @langchain/anthropic @langchain/core
 *   export ANTHROPIC_API_KEY=sk-ant-...
 *   node 15_langgraph_multiagent.js
 */

import { StateGraph, Annotation, END, START } from '@langchain/langgraph';
import { ChatAnthropic } from '@langchain/anthropic';
import { HumanMessage, AIMessage } from '@langchain/core/messages';

// ─────────────────────────────────────────────
// 初始化模型
// ─────────────────────────────────────────────

// 所有 Worker 共用同一个模型（也可以给不同 Worker 用不同模型）
const model = new ChatAnthropic({
  model: 'claude-opus-4-5',
  maxTokens: 2048,
  apiKey: process.env.ANTHROPIC_API_KEY,
});

// ─────────────────────────────────────────────
// 1. 共享 State 定义
// ─────────────────────────────────────────────

/**
 * 多 Agent 系统的共享状态（所有节点读写同一个 State）
 *
 * 类比：团队共享的项目看板
 *   - messages:      所有 Agent 的发言记录（追加）
 *   - next:          Supervisor 的路由决策（替换）
 *   - task:          原始任务描述（替换）
 *   - researchNotes: researcher 的调研结果（替换）
 *   - code:          coder 生成的代码（替换）
 *   - reviewResult:  reviewer 的审查结论（替换）
 *   - finalOutput:   最终交付内容（替换）
 *   - iterationCount: 已迭代次数，防止死循环（累加）
 */
const MultiAgentState = Annotation.Root({
  // 消息历史（追加，不覆盖）
  messages: Annotation({
    reducer: (existing, newMsgs) => {
      if (!newMsgs) return existing;
      const msgs = Array.isArray(newMsgs) ? newMsgs : [newMsgs];
      return [...existing, ...msgs];
    },
    default: () => [],
  }),

  // Supervisor 决定的下一节点
  next: Annotation({
    reducer: (_, newVal) => newVal,
    default: () => 'supervisor',
  }),

  // 原始任务描述
  task: Annotation({
    reducer: (old, newVal) => newVal ?? old,
    default: () => '',
  }),

  // researcher 的调研笔记
  researchNotes: Annotation({
    reducer: (old, newVal) => newVal ?? old,
    default: () => '',
  }),

  // coder 写的代码
  code: Annotation({
    reducer: (old, newVal) => newVal ?? old,
    default: () => '',
  }),

  // reviewer 的审查结论
  reviewResult: Annotation({
    reducer: (old, newVal) => newVal ?? old,
    default: () => '',
  }),

  // 最终交付物
  finalOutput: Annotation({
    reducer: (old, newVal) => newVal ?? old,
    default: () => '',
  }),

  // 迭代计数器（防止无限循环）
  iterationCount: Annotation({
    reducer: (old, newVal) => (newVal !== undefined ? newVal : old),
    default: () => 0,
  }),
});

// ─────────────────────────────────────────────
// 2. Supervisor Agent（路由决策者）
// ─────────────────────────────────────────────

/**
 * Supervisor 的职责：
 *   读取当前 State，决定下一步交给哪个 Worker，
 *   当任务完成或超出迭代限制时宣布 FINISH。
 *
 * 路由规则（基于 State 字段）：
 *   无 researchNotes                   → researcher
 *   有 researchNotes，无 code          → coder
 *   有 code，无 reviewResult           → reviewer
 *   reviewResult 包含 "PASS"           → FINISH
 *   reviewResult 不包含 "PASS" 且 < 3次 → coder（修改代码）
 *   超过最大迭代次数                   → FINISH（强制结束）
 */
async function supervisorNode(state) {
  console.log('\n[Supervisor] 分析进展，决定路由...');

  const MAX_ITERATIONS = 4;

  if (state.iterationCount >= MAX_ITERATIONS) {
    console.log(`[Supervisor] 已达最大迭代次数（${MAX_ITERATIONS}），强制结束`);
    return {
      next: 'FINISH',
      finalOutput: state.code || '超出迭代限制',
      messages: [new AIMessage(`[Supervisor] 强制结束（已迭代 ${state.iterationCount} 次）`)],
    };
  }

  let nextNode;
  let reasoning;

  if (!state.researchNotes) {
    nextNode = 'researcher';
    reasoning = '尚无调研资料，先让 researcher 搜集背景信息';
  } else if (!state.code) {
    nextNode = 'coder';
    reasoning = '调研完成，让 coder 开始写代码';
  } else if (!state.reviewResult) {
    nextNode = 'reviewer';
    reasoning = '代码写好了，让 reviewer 进行审查';
  } else if (state.reviewResult.startsWith('PASS') || state.reviewResult.includes('通过')) {
    nextNode = 'FINISH';
    reasoning = '审查通过，任务完成！';
  } else {
    nextNode = 'coder';
    reasoning = '审查未通过，让 coder 根据反馈修改代码';
  }

  console.log(`[Supervisor] → ${nextNode}（${reasoning}）`);

  return {
    next: nextNode,
    iterationCount: state.iterationCount + 1,
    messages: [new AIMessage(`[Supervisor] ${reasoning} → ${nextNode}`)],
  };
}

// ─────────────────────────────────────────────
// 3. Worker：researcher（调研员）
// ─────────────────────────────────────────────

/**
 * researcher 的职责：
 *   搜集与任务相关的背景知识、最佳实践、注意事项，
 *   输出结构化调研报告，供 coder 参考。
 */
async function researcherNode(state) {
  console.log('\n[Researcher] 开始调研...');

  const response = await model.invoke([
    new HumanMessage(
      `你是技术调研员。请为以下编程任务提供简洁的背景知识和最佳实践（150字以内）：
关键概念、常见陷阱、推荐实现方式各列1-2条。

任务：${state.task}`
    ),
  ]);

  const researchNotes = response.content;
  console.log('[Researcher] 调研完成');

  return {
    researchNotes,
    messages: [new AIMessage(`[Researcher 调研报告]\n${researchNotes}`)],
  };
}

// ─────────────────────────────────────────────
// 4. Worker：coder（开发员）
// ─────────────────────────────────────────────

/**
 * coder 的职责：
 *   根据任务描述和调研报告写代码；
 *   若有 reviewer 反馈，根据反馈修改代码。
 */
async function coderNode(state) {
  const isRevision = !!state.reviewResult && !state.reviewResult.startsWith('PASS');
  console.log(`\n[Coder] ${isRevision ? '修改代码（根据审查反馈）' : '初次编写代码'}...`);

  const prompt = isRevision
    ? `任务：${state.task}

调研报告：
${state.researchNotes}

当前代码：
\`\`\`javascript
${state.code}
\`\`\`

审查反馈（请据此修改）：
${state.reviewResult}

请输出修改后的完整代码，只输出代码本身，不要额外说明。`
    : `任务：${state.task}

调研报告（请参考）：
${state.researchNotes}

请用 JavaScript/Node.js 实现，代码简洁可读，有必要的注释。只输出代码本身。`;

  const response = await model.invoke([
    new HumanMessage(prompt),
  ]);

  console.log('[Coder] 代码编写完成');

  return {
    code: response.content,
    reviewResult: '',  // 清空上一次审查结果，等待新审查
    messages: [new AIMessage(`[Coder ${isRevision ? '修改版' : '初版'}]\n${response.content}`)],
  };
}

// ─────────────────────────────────────────────
// 5. Worker：reviewer（审查员）
// ─────────────────────────────────────────────

/**
 * reviewer 的职责：
 *   检查代码是否符合任务要求、有无明显 Bug、代码风格是否良好。
 *   通过 → 回复以 "PASS" 开头；
 *   不通过 → 回复以 "FAIL" 开头，列出具体问题。
 */
async function reviewerNode(state) {
  console.log('\n[Reviewer] 开始代码审查...');

  const response = await model.invoke([
    new HumanMessage(
      `你是代码审查员。请检查以下代码：
1. 是否完成任务要求
2. 有无明显 Bug 或逻辑错误
3. 代码可读性是否良好
4. 有无必要的错误处理

通过则回复以 "PASS" 开头；有问题则以 "FAIL" 开头列出问题（100字以内）。

任务：${state.task}

代码：
\`\`\`javascript
${state.code}
\`\`\``
    ),
  ]);

  const reviewResult = response.content;
  const isPassing = reviewResult.startsWith('PASS') || reviewResult.includes('通过');
  console.log(`[Reviewer] 审查结果：${isPassing ? 'PASS ✅' : 'FAIL ❌'}`);

  return {
    reviewResult,
    finalOutput: isPassing ? state.code : state.finalOutput,
    messages: [new AIMessage(`[Reviewer 审查结论]\n${reviewResult}`)],
  };
}

// ─────────────────────────────────────────────
// 6. 条件路由函数
// ─────────────────────────────────────────────

/**
 * 根据 Supervisor 设置的 next 字段决定走哪条路
 * 这是"条件边"的路由函数
 */
function routeFromSupervisor(state) {
  return state.next; // "researcher" | "coder" | "reviewer" | "FINISH"
}

// ─────────────────────────────────────────────
// 7. 构建多 Agent 图（Star 拓扑）
// ─────────────────────────────────────────────

/**
 * Star 拓扑：所有 Worker 通过 Supervisor 协调
 *
 * 图结构：
 *   START → supervisor
 *   supervisor → [条件边] → researcher / coder / reviewer / FINISH(END)
 *   researcher → supervisor
 *   coder      → supervisor
 *   reviewer   → supervisor
 *
 * Mermaid 可视化（粘贴到 https://mermaid.live）：
 *   graph TD
 *     start([开始]) --> supervisor[Supervisor]
 *     supervisor -->|researcher| researcher[Researcher 调研员]
 *     supervisor -->|coder| coder[Coder 开发员]
 *     supervisor -->|reviewer| reviewer[Reviewer 审查员]
 *     supervisor -->|FINISH| end_([结束])
 *     researcher --> supervisor
 *     coder --> supervisor
 *     reviewer --> supervisor
 */
function buildMultiAgentGraph() {
  const workflow = new StateGraph(MultiAgentState);

  // 添加所有节点
  workflow.addNode('supervisor', supervisorNode);
  workflow.addNode('researcher', researcherNode);
  workflow.addNode('coder',      coderNode);
  workflow.addNode('reviewer',   reviewerNode);

  // 入口：START → supervisor
  workflow.addEdge(START, 'supervisor');

  // supervisor 的条件出边
  workflow.addConditionalEdges(
    'supervisor',
    routeFromSupervisor,
    {
      researcher: 'researcher',
      coder:      'coder',
      reviewer:   'reviewer',
      FINISH:     END,  // 特殊值 FINISH → 图结束
    }
  );

  // 每个 Worker 完成后回到 supervisor（由它决定下一步）
  workflow.addEdge('researcher', 'supervisor');
  workflow.addEdge('coder',      'supervisor');
  workflow.addEdge('reviewer',   'supervisor');

  return workflow.compile();
}

// ─────────────────────────────────────────────
// 8. 完整演示
// ─────────────────────────────────────────────

async function demo_fullPipeline() {
  console.log('\n=== 完整演示：多 Agent 协作完成编程任务 ===');
  console.log('流程：supervisor → researcher → coder → reviewer → (修改?) → 最终交付\n');

  const graph = buildMultiAgentGraph();

  const task = '实现一个 JavaScript 函数：计算数组中所有数字的平均值，要求处理空数组和非数字元素的边界情况';

  console.log('任务:', task);
  console.log('='.repeat(60));

  const finalState = await graph.invoke({
    task,
    messages: [new HumanMessage(task)],
  });

  // 打印最终结果
  console.log('\n' + '='.repeat(60));
  console.log('【多 Agent 协作完成！最终交付物】');
  console.log('='.repeat(60));

  if (finalState.researchNotes) {
    console.log('\n调研报告（前100字）:');
    console.log(finalState.researchNotes.slice(0, 100) + '...');
  }

  if (finalState.code) {
    console.log('\n最终代码:');
    console.log(finalState.code);
  }

  if (finalState.reviewResult) {
    console.log('\n审查结论:');
    console.log(finalState.reviewResult.slice(0, 120));
  }

  console.log(`\n完成！Supervisor 共做了 ${finalState.iterationCount} 次路由决策`);
  console.log(`对话消息总数: ${finalState.messages.length} 条`);
}

// ─────────────────────────────────────────────
// 主函数
// ─────────────────────────────────────────────

async function main() {
  console.log('╔═══════════════════════════════════════════════╗');
  console.log('║  15_langgraph_multiagent.js  多 Agent 协作    ║');
  console.log('╚═══════════════════════════════════════════════╝');
  console.log(`
架构（Star 拓扑）：
          ┌─────────────┐
          │  Supervisor  │ ← 路由决策，分配任务
          └──────┬───────┘
                 │ 条件边（根据 next 字段）
    ┌────────────┼────────────┐
    ▼            ▼            ▼
researcher     coder       reviewer
  调研员        开发员        审查员
    └────────────┴────────────┘
                 │ 完成后回到 supervisor
`);

  try {
    await demo_fullPipeline();
  } catch (error) {
    console.error('演示出错:', error.message);
    if (error.message.includes('Cannot find package')) {
      console.error('\n请先安装依赖:');
      console.error('  npm install @langchain/langgraph @langchain/anthropic @langchain/core');
    } else {
      console.error(error.stack);
    }
  }

  console.log('\n演示完成。');
}

main().catch(err => {
  console.error('未捕获异常:', err.message);
  process.exit(1);
});
