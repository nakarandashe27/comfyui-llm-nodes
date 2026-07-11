// Бейдж фактической стоимости последней генерации (nodes-image-v2.md §5.2).
// Python-нода кладёт cost_usd в ui-поле результата -> onExecuted -> read-only виджет.
import { app } from "../../scripts/app.js";

const LLM_NODES = ["LLMText", "LLMImage", "LLMVideo"];

app.registerExtension({
  name: "comfyui-llm-nodes.costBadge",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!LLM_NODES.includes(nodeData.name)) return;
    const orig = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      orig?.apply(this, arguments);
      const cost = message?.cost_usd?.[0];
      if (cost == null) return; // шлюз не прислал заголовок — молчим, не рисуем нули
      const label = "$" + Number(cost).toFixed(4);
      let w = this.widgets?.find((x) => x.name === "last_cost");
      if (!w) {
        w = this.addWidget("text", "last_cost", label, () => {});
        w.serialize = false; // не сохранять в воркфлоу
      }
      w.value = label;
      this.setDirtyCanvas(true, false);
    };
  },
});
