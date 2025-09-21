"""
Enhanced AgentManager for the develop branch.

This implementation wraps the new LangGraph based routing (and optional
planning) pipeline defined in ``agents.graphs.pipeline_graph``.  It
serves as a drop‐in replacement for the previous stage‑1 manager,
offering the same high‑level ``plan`` method but leveraging the
underlying graph to perform intent classification, tool selection and
optional planning in a single call.

Key differences from the old implementation:

* The manager defers all routing logic to the LangGraph pipeline
  defined in ``agents.graphs.pipeline_graph``.  This removes the
  explicit dependency on ``MainRouterAgent`` and ``QueryRouter``.  The
  pipeline internally handles main routing, optional query routing
  (depending on the ``run_mode``), fusing results and generating plans.
* Configuration values (e.g. run mode, scoring thresholds) are
  translated into graph parameters.  If a capability map is provided
  via the embedding config file, it will be used to filter allowed
  user‑facing tools.
* The ``plan`` method now returns a richer structure including the
  intermediate route and needs, the list of query router candidates,
  and the optional planning output.  This affords downstream
  components greater flexibility to introspect decisions and act on
  plans.  To maintain backward compatibility, the top‑level keys
  ``final_route`` and ``tools`` remain unchanged.

If ``upgrade_to_complex_if_multi_need`` is enabled and the model
predicts a simple route while producing two or more needs, the
manager will upgrade the final route to ``COMPLEX_TOOL`` and expose
all unique needs as tools.  This mirrors the consistency override
behaviour of the original manager.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

# Import the unified routing→planning pipeline.  It is assumed that
# ``agents.graphs.pipeline_graph`` exists in the develop branch and
# exposes a ``run_agent`` convenience function.  When invoked, it
# returns a dictionary containing keys such as ``final_route``,
# ``tools``, ``plan``, ``route``, ``needs``, ``qr_candidates`` and
# ``debug``.
try:
    from agents.graphs.pipeline_graph import run_agent as _run_agent_pipeline  # type: ignore
except Exception as e:  # pragma: no cover
    # Fallback for environments where the pipeline is not available.  In
    # such cases, developers should ensure that the appropriate graph
    # modules are on the PYTHONPATH.  A NotImplementedError will be
    # raised when attempting to plan.
    _run_agent_pipeline = None  # type: ignore


@dataclass
class ManagerConfig:
    """
    Configuration options for AgentManager.

    Attributes
    ----------
    run_mode: str
        Orchestration mode for routing.  ``"fusion"`` runs both the
        main router and query router; ``"main_only"`` skips the query
        router and derives tools directly from the main router's needs.
    embedding_config_path: str
        Path to the embedding configuration JSON.  When provided, the
        manager sets ``QUERY_ROUTER_CONFIG`` in the environment and
        attempts to derive a set of user‑facing tools from the
        capability map within the config.  This set is passed to the
        graph as ``allowed_capabilities``.
    top_k: int
        Maximum number of tools to return.  This maps to the
        ``max_tools`` parameter of the routing graph.
    score_floor: float
        Minimum score threshold for query router candidates.  This
        maps to ``min_qr_score`` in the routing graph.
    upgrade_to_complex_if_multi_need: bool
        Whether to override a ``SIMPLE_TOOL`` route with
        ``COMPLEX_TOOL`` when two or more needs are predicted.  This
        mirrors the original manager's behaviour.
    enable_debug_logging: bool
        If true, the manager logs detailed diagnostic information.
    """

    run_mode: str = "fusion"
    embedding_config_path: str = ""
    top_k: int = 3
    score_floor: float = 0.0
    upgrade_to_complex_if_multi_need: bool = True
    enable_debug_logging: bool = True

    def __post_init__(self) -> None:
        # Normalize the run mode and default to fusion on invalid input
        self.run_mode = (self.run_mode or "fusion").lower().strip()
        if self.run_mode not in ("fusion", "main_only"):
            self.run_mode = "fusion"


def _setup_logger(enable_debug: bool = True) -> logging.Logger:
    """Configure and return a logger for the manager."""
    logger = logging.getLogger("AgentManager")
    if not logger.handlers:
        level_name = os.getenv(
            "AGENTMGR_LOGLEVEL", "DEBUG" if enable_debug else "INFO"
        )
        level = getattr(logging, level_name.upper(),
                        logging.DEBUG if enable_debug else logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    logger.setLevel(logging.DEBUG if enable_debug else logging.INFO)
    return logger


class AgentManager:
    """
    Stage‑1+ manager built atop LangGraph.

    This class exposes a ``plan`` method that returns a dict
    containing the final route, selected tools and auxiliary
    information.  Internally it delegates all routing and (optional)
    planning to the LangGraph pipeline defined in
    ``agents.graphs.pipeline_graph``.
    """

    def __init__(self, cfg: Optional[ManagerConfig] = None) -> None:
        self.cfg = cfg or ManagerConfig()
        self.log = _setup_logger(self.cfg.enable_debug_logging)

        self.log.info(
            "Initializing AgentManager (graph-based) ... mode=%s", self.cfg.run_mode)

        # Set up environment variable so that QueryRouterNode can locate the
        # embedding config when running in fusion mode.  This mirrors the
        # behaviour of the original manager.  Note that if the file does
        # not exist, the query router may fail and be gracefully skipped.
        if self.cfg.embedding_config_path:
            os.environ["QUERY_ROUTER_CONFIG"] = self.cfg.embedding_config_path
            self.log.debug("Set QUERY_ROUTER_CONFIG=%s",
                           self.cfg.embedding_config_path)

        # Attempt to derive allowed user‑facing tools from the capability
        # map.  The config file is expected to contain a ``capability``
        # dictionary mapping card names to metadata.  Only cards where
        # ``is_support`` is false are considered user‑facing.  If the
        # file cannot be loaded or does not contain such a map, the
        # graph will fall back to its built‑in allowed set.
        self.allowed_caps: Optional[Set[str]] = None
        if self.cfg.embedding_config_path:
            try:
                with open(self.cfg.embedding_config_path, "r", encoding="utf-8") as f:
                    cfg_raw = json.load(f)
                cap_map = cfg_raw.get("capability", {})
                allowed: Set[str] = set()
                for card, meta in cap_map.items():
                    if not isinstance(meta, dict) or not meta.get("is_support", False):
                        allowed.add(card)
                if allowed:
                    self.allowed_caps = allowed
                    self.log.info(
                        "Loaded %d allowed capabilities from config.", len(allowed))
            except Exception as e:
                self.log.warning(
                    "Could not load capability map from '%s': %s", self.cfg.embedding_config_path, e
                )

        # Verify that the pipeline function is available
        if _run_agent_pipeline is None:  # pragma: no cover
            self.log.error(
                "agents.graphs.pipeline_graph.run_agent could not be imported. "
                "Ensure that the develop branch graph modules are accessible."
            )

        self.log.info("AgentManager ready.")

    # ----- internal -----
    def _graph_params(self) -> Dict[str, Any]:
        """Map ManagerConfig into parameters for the routing pipeline."""
        params: Dict[str, Any] = {
            "mode": self.cfg.run_mode,
            "min_qr_score": self.cfg.score_floor,
            "max_tools": self.cfg.top_k,
            # preserve the original selection logic: 1 primary tool for
            # simple routes, 2 or more for complex routes
            "simple_max_primary": 1,
            "complex_min_primary": 2,
            # always require user‑facing tools; this prevents support tools
            # from being surfaced to the user
            "require_user_facing": True,
        }
        if self.allowed_caps:
            params["allowed_capabilities"] = self.allowed_caps
        return params

    # ----- public API -----
    def plan(self, user_text: str, memory: Any = None) -> Dict[str, Any]:
        """
        Determine the agent's high‑level intent and select appropriate tools.

        Parameters
        ----------
        user_text: str
            Raw user input.
        memory: Any, optional
            Conversation memory or context object.  This is passed
            through to the underlying nodes unchanged.

        Returns
        -------
        Dict[str, Any]
            A structure containing at least the keys ``final_route``
            and ``tools``.  Additional keys include ``plan``, ``route``,
            ``needs``, ``qr_candidates`` and ``debug``.
        """
        if not _run_agent_pipeline:
            raise NotImplementedError(
                "The routing/planning pipeline is unavailable. "
                "Check your installation of agents.graphs.pipeline_graph."
            )

        self.log.info("=== PLAN (%s) START ===", self.cfg.run_mode)
        self.log.debug("User text: %s", user_text)

        # Invoke the LangGraph pipeline
        params = self._graph_params()
        try:
            res: Dict[str, Any] = _run_agent_pipeline(
                user_text, memory=memory, **params)
        except Exception as e:
            # In the event of any exception, fall back to GENERAL_CHAT
            # with no tools.  The error is surfaced in the debug field.
            self.log.exception("Pipeline invocation failed: %s", e)
            return {
                "final_route": "GENERAL_CHAT",
                "tools": [],
                "debug": {"error": str(e)},
            }

        # Optionally override SIMPLE→COMPLEX if multiple needs are predicted
        needs: List[str] = list(res.get("needs") or [])
        final_route: str = res.get("final_route") or "GENERAL_CHAT"
        if (
            self.cfg.upgrade_to_complex_if_multi_need
            and final_route == "SIMPLE_TOOL"
            and len(needs) >= 2
        ):
            self.log.info(
                "Override route -> COMPLEX_TOOL (needs >= 2 and simple predicted)"
            )
            final_route = "COMPLEX_TOOL"
            # Derive tools by taking all unique needs
            seen: Set[str] = set()
            tools: List[str] = []
            for need in needs:
                if need not in seen:
                    tools.append(need)
                    seen.add(need)
            res["tools"] = tools
            res["final_route"] = final_route

        # Compose final output.  Expose additional fields for
        # transparency while keeping the core contract of ``final_route``
        # and ``tools`` intact.
        out: Dict[str, Any] = {
            "final_route": final_route,
            "tools": res.get("tools", []),
            # Pass through the optional planning output.  Consumers may
            # ignore this field if they only need routing.
            "plan": res.get("plan"),
            # Include intermediate routing info for debugging/introspection
            "route": res.get("route"),
            "needs": needs,
            "qr_candidates": res.get("qr_candidates", []),
            "debug": res.get("debug", {}),
        }

        self.log.debug("Plan result: %s", out)
        self.log.info("=== PLAN (%s) END ===", self.cfg.run_mode)
        return out


# Allow external modules to import the manager class and config easily
__all__ = [
    "ManagerConfig",
    "AgentManager",
]
