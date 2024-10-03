from absl import app
import mujoco
from mujoco import mjx


def main(args: None):
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    data = mujoco.MjData(model)
    model_mjx = mjx.put_model(model)
    data_mjx = mjx.make_data(model)

    print(data_mjx.qpos)
    print(data.qpos)


if __name__ == "__main__":
    app.run(main)
