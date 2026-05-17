/**
 * Local assistant replies when GPU /api/assistant is unavailable.
 */
export function getAssistantResponse({ command, parsed }) {
  const intent = parsed?.intent;
  const ent = parsed?.entities || {};

  let reply = "";
  switch (intent) {
    case "OPEN_MAIN_DASHBOARD":
      reply = "Yes boss! Dashboard is coming up right away.";
      break;
    case "OPEN_PACKAGE_VIEW":
      reply = `Yes boss! Opening ${ent.package_name || "that package"} now.`;
      break;
    case "FILTER_DASHBOARD_BY_BANK_DATE":
      reply = `Yes boss! Locking the command center to ${ent.bank_name || "that bank"} on ${ent.date || "that date"}. All packages are updating.`;
      break;
    case "FILTER_PACKAGE_BY_BANK_DATE":
      reply = `Yes boss! ${ent.package_name || "That view"} for ${ent.bank_name || ""} on ${ent.date || ""} — pulling it up.`;
      break;
    case "RESET_FILTERS":
      reply = "Yes boss! Filters are cleared. Back to the full rail view.";
      break;
    case "UNKNOWN_COMMAND":
      reply =
        "Sorry boss, I could not match that command. Try open the dashboard, or a bank name with a date.";
      break;
    default:
      if (parsed?.clarification) {
        reply = `Boss — ${parsed.clarification}`;
      } else if (command) {
        reply = `Yes boss! Working on: ${command}`;
      } else {
        reply = "How can I help, boss?";
      }
  }

  return { reply };
}
