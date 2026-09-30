"""Project-specific RSL-RL runner overrides."""

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner


class KidRLOnPolicyRunner(VelocityOnPolicyRunner):
  """VelocityOnPolicyRunner, but resume never restores common_step_counter.

  MjlabOnPolicyRunner.load() restores env.common_step_counter from the
  checkpoint "to preserve curricula state" -- but step_based_staged_curriculum
  (see mdp.py) tracks its own progress as a plain `current_stage` attribute on
  a fresh Python object, which is NOT saved/restored across resume and always
  starts back at 0. Restoring common_step_counter alone (without current_stage
  keeping pace) makes the curriculum fast-forward through every already-passed
  stage threshold within the first few post-resume steps instead of the
  intended gradual per-iteration ramp.

  Simpler fix than persisting current_stage too: just don't restore
  common_step_counter either, so both start at 0 together on every resume and
  stay in sync. The tradeoff is that step-based curriculum/event thresholds
  are counted from each resume's start rather than total lifetime training,
  but that's a much smaller price than the fast-forward cascade.
  """

  def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None):
    infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
    self.env.unwrapped.common_step_counter = 0
    return infos
