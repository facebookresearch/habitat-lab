# XR reader application

A simple HITL application that displays XR input from a remote client.

## Client Application

This is an experimental server. The client application is currently unavailable.

## Setup

### Quest

#### Requirements:

* Quest headset (preferably Quest 3)
* Meta account
* Device set up in [Developer Mode](https://developers.meta.com/horizon/documentation/native/android/mobile-device-setup/)
* A client capable of installing applications on Quest.
  * On MacOS, [Meta Quest Developer Hub](https://developers.meta.com/horizon/documentation/unity/ts-odh/) is recommended.
  * [SideQuest](https://sidequestvr.com/) is also an option.
* Have both the headset and server on the same network.

#### Setup:

1. Install the client to the headset.
2. Launch the server. See the launch command below.
3. Launch the application on the headset.

The headset should automatically discover and connect to the server.

#### Troubleshooting:

If the headset cannot connect to the server, check the following items:

1. Make sure that both devices are on the same network.
   * Otherwise, make sure that the devices can communicate to each other. See the step below.
2. If the server has a firewall, make sure that the following ports are open:
   * Discovery: `UDP` on `12345`.
   * Connection: `TCP` on `18000`.

## Launch Command

Run the following command from the root `habitat-lab` directory:

```bash
python examples/hitl/experimental/xr_reader/xr_reader.py
```
